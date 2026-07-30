import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import pandas as pd

# ── Data Loading ────────────────────────────────────────────────────────────────
df_imaging_merged = pd.read_csv("imaging_icd_demo.csv")
df_imaging_merged["CALC_DX_DATE"] = pd.to_datetime(df_imaging_merged["CALC_DX_DATE"])
df_imaging_merged.to_csv("imaging_icd_demo.csv", index=False)
print(f"Total imaging rows: {len(df_imaging_merged)}")

df_card_imaging = df_imaging_merged[
    df_imaging_merged.PROC_NAME.str.contains(
        "coronary|echo|angio|card", case=False
    )
]
print(f"Cardiac imaging rows: {len(df_card_imaging)}")

# ── Configuration ────────────────────────────────────────────────────────────────
BASE_URL                = "http://localhost:11424"
CHAT_URL                = f"{BASE_URL}/api/chat"
HEALTH_URL              = f"{BASE_URL}/healthz"
MODEL                   = "google/gemma-4-12B-it"
SAVE_PATH               = Path("takotsubo_results_cards.csv")
ERROR_PATH              = Path("takotsubo_errors_cards.csv")
ID_COL                  = "ORDER_ID"
CLIENT_CONCURRENCY      = int(os.getenv("CLIENT_CONCURRENCY", "8"))
CHECKPOINT_SIZE         = 100
MAX_RETRIES             = 5
REQUEST_TIMEOUT_SECONDS = 360
VALID_CLASSIFICATIONS   = {"Confirmed", "Possible", "Absent"}

# ── System Prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert cardiologist.

Review the clinical report and determine whether there is evidence of
Takotsubo cardiomyopathy.

Consider:
- Takotsubo cardiomyopathy
- Stress cardiomyopathy
- Apical ballooning syndrome
- Imaging findings consistent with Takotsubo

Return ONLY valid JSON with exactly these fields:

{
  "classification": "Confirmed" | "Possible" | "Absent",
  "evidence": "<short direct quote, or empty string>",
  "reason": "<one concise sentence>"
}
""".strip()


# ── Helpers ──────────────────────────────────────────────────────────────────────
def load_processed_ids() -> set[str]:
    if not SAVE_PATH.exists():
        return set()
    saved = pd.read_csv(SAVE_PATH, dtype={ID_COL: str})
    if ID_COL not in saved.columns:
        return set()
    if "classification" in saved.columns:
        saved = saved[saved["classification"].isin(VALID_CLASSIFICATIONS)]
    return set(saved[ID_COL].astype(str))


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame[ID_COL] = frame[ID_COL].astype(str)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


# ── Terminal Output ───────────────────────────────────────────────────────────────
async def terminal_log(output_lock: asyncio.Lock, message: str) -> None:
    """Print one complete message without interleaving concurrent workers."""
    async with output_lock:
        print(message, flush=True)


# ── Server Capacity ───────────────────────────────────────────────────────────────
async def get_server_capacity(session: aiohttp.ClientSession) -> int:
    async with session.get(HEALTH_URL) as response:
        response.raise_for_status()
        health = await response.json()

    if not health.get("loaded"):
        raise RuntimeError("llm-serve is running but the model is not loaded")

    tuning   = health.get("throughput_tuning") or {}
    capacity = health.get("foreground_capacity")

    print("\n── Server tuning profile ──────────────────────────────")
    print(json.dumps(tuning, indent=2))
    print(f"foreground_capacity : {capacity}")
    print("────────────────────────────────────────────────────────\n")

    if not isinstance(capacity, int) or capacity < 1:
        raise RuntimeError(
            f"Invalid foreground capacity in /healthz: {capacity!r}"
        )
    return capacity


# ── Validation ────────────────────────────────────────────────────────────────────
def validate_result(raw_text: str) -> dict:
    # Strip markdown code fences if the model wraps its JSON
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    result = json.loads(text)

    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object")

    classification = result.get("classification")
    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(f"Invalid classification: {classification!r}")

    evidence = result.get("evidence", "")
    reason   = result.get("reason", "")

    if not isinstance(evidence, str):
        raise ValueError("evidence must be a string")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    return {
        "classification": classification,
        "evidence":       evidence.strip(),
        "reason":         reason.strip(),
    }


# ── Core Classifier ───────────────────────────────────────────────────────────────
async def classify_report(
    session:    aiohttp.ClientSession,
    current_id: str,
    report:     str,
    stats:      dict,
    stats_lock: asyncio.Lock,
    output_lock: asyncio.Lock,
) -> dict:
    payload = {
        "model":    MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": report},
        ],
        "stream":  False,
        "options": {
            "temperature": 0,
            "top_p":       1,
            "num_predict": 128,
        },
    }

    last_error = None
    request_started = time.perf_counter()

    for attempt in range(MAX_RETRIES):
        attempt_number = attempt + 1
        await terminal_log(
            output_lock,
            f"[{datetime.now():%H:%M:%S}] SENDING "
            f"order={current_id} attempt={attempt_number}/{MAX_RETRIES}",
        )
        try:
            async with session.post(CHAT_URL, json=payload) as response:
                if response.status in {202, 429, 502, 503, 504}:
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after) if retry_after
                        else min(2 ** attempt, 30)
                    )
                    delay += random.uniform(0, 0.5)

                    async with stats_lock:
                        stats["retries"] += 1

                    await terminal_log(
                        output_lock,
                        f"[{datetime.now():%H:%M:%S}] RETRY "
                        f"order={current_id} status={response.status} "
                        f"sleep={delay:.1f}s",
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                data = await response.json(content_type=None)

            message = data.get("message")
            if not isinstance(message, dict):
                raise ValueError("Response does not contain a message object")

            response_text = message.get("content")
            if not isinstance(response_text, str):
                raise ValueError("Response message content is not a string")

            latency = time.perf_counter() - request_started
            await terminal_log(
                output_lock,
                (
                    f"\n[{datetime.now():%H:%M:%S}] LLM RESPONSE "
                    f"order={current_id} attempt={attempt_number} "
                    f"latency={latency:.2f}s\n"
                    f"{response_text}\n"
                    f"{'-' * 80}"
                ),
            )

            result = validate_result(response_text)
            result[ID_COL] = current_id
            result["_latency_seconds"] = latency
            result["_attempts"] = attempt_number
            return result

        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc
            await terminal_log(
                output_lock,
                f"[{datetime.now():%H:%M:%S}] REQUEST ERROR "
                f"order={current_id} attempt={attempt_number}: {exc}",
            )
            if attempt + 1 < MAX_RETRIES:
                async with stats_lock:
                    stats["retries"] += 1
                await asyncio.sleep(
                    min(2 ** attempt, 30) + random.uniform(0, 0.5)
                )

    return {
        ID_COL: current_id,
        "error": str(last_error),
        "_latency_seconds": time.perf_counter() - request_started,
        "_attempts": MAX_RETRIES,
    }


# ── Worker ────────────────────────────────────────────────────────────────────────
async def worker(
    queue:        asyncio.Queue,
    result_queue: asyncio.Queue,
    session:      aiohttp.ClientSession,
    stats:        dict,
    stats_lock:   asyncio.Lock,
    output_lock:  asyncio.Lock,
) -> None:
    while True:
        job = await queue.get()
        try:
            if job is None:
                return
            current_id, report = job
            try:
                result = await classify_report(
                    session,
                    current_id,
                    report,
                    stats,
                    stats_lock,
                    output_lock,
                )
            except Exception as exc:
                result = {
                    ID_COL: current_id,
                    "error": f"Unexpected worker error: {exc}",
                    "_latency_seconds": 0.0,
                    "_attempts": 0,
                }
            await result_queue.put(result)
        finally:
            queue.task_done()


async def feed_jobs(
    queue: asyncio.Queue,
    jobs: list[tuple[str, str]],
    worker_count: int,
) -> None:
    """Feed the bounded queue without blocking result collection."""
    for job in jobs:
        await queue.put(job)
    for _ in range(worker_count):
        await queue.put(None)


# ── Main ──────────────────────────────────────────────────────────────────────────
async def main() -> None:
    processed_ids = load_processed_ids()

    jobs: list[tuple[str, str]] = []
    for row in df_card_imaging.itertuples(index=False):
        row_data   = row._asdict()
        current_id = str(row_data[ID_COL])
        if current_id in processed_ids:
            continue
        report = (
            "ORDER IMPRESSION:\n"
            f"{row_data.get('ORDER_IMPRESSION', '')}\n\n"
            "ORDER NARRATIVE:\n"
            f"{row_data.get('ORDER_NARRATIVE', '')}"
        )
        jobs.append((current_id, report))

    total = len(jobs)
    print(f"Already processed  : {len(processed_ids)}")
    print(f"New reports to run : {total}")

    if not total:
        print("Nothing to do — exiting.")
        return

    # ── Stats dict (shared, lock-protected) ──────────────────────────────────
    stats: dict = {
        "processed":  0,
        "confirmed":  0,
        "possible":   0,
        "absent":     0,
        "errors":     0,
        "retries":    0,
    }
    stats_lock  = asyncio.Lock()
    output_lock = asyncio.Lock()
    start_time  = time.time()

    # ── Probe server capacity ────────────────────────────────────────────────
    timeout = aiohttp.ClientTimeout(
        total=REQUEST_TIMEOUT_SECONDS, connect=30,
        sock_read=REQUEST_TIMEOUT_SECONDS,
    )
    async with aiohttp.ClientSession(timeout=timeout) as probe:
        tuned_capacity = await get_server_capacity(probe)

    concurrency = CLIENT_CONCURRENCY or tuned_capacity
    concurrency = min(concurrency, tuned_capacity)

    print(
        f"Concurrency        : {concurrency}  "
        f"(server tuned capacity: {tuned_capacity})"
    )
    print(f"Checkpoint every   : {CHECKPOINT_SIZE} rows")
    print(f"Max retries        : {MAX_RETRIES}")
    print()
    print("Starting — every request and LLM response will be printed.\n")

    connector = aiohttp.TCPConnector(
        limit=concurrency, limit_per_host=concurrency,
        keepalive_timeout=60, enable_cleanup_closed=True,
    )

    job_queue    = asyncio.Queue(maxsize=concurrency * 2)
    result_queue = asyncio.Queue()

    successful_buffer: list[dict] = []
    error_buffer:      list[dict] = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Start workers
        workers = [
            asyncio.create_task(
                worker(
                    job_queue,
                    result_queue,
                    session,
                    stats,
                    stats_lock,
                    output_lock,
                )
            )
            for _ in range(concurrency)
        ]

        # Feed jobs concurrently so completed results are collected immediately.
        producer_task = asyncio.create_task(
            feed_jobs(job_queue, jobs, len(workers))
        )

        # Collect results
        for _ in range(total):
            result = await result_queue.get()

            try:
                latency = float(result.pop("_latency_seconds", 0.0))
                attempts = int(result.pop("_attempts", 0))
                async with stats_lock:
                    stats["processed"] += 1
                    clf = result.get("classification", "")
                    if clf == "Confirmed":
                        stats["confirmed"] += 1
                    elif clf == "Possible":
                        stats["possible"]  += 1
                    elif clf == "Absent":
                        stats["absent"]    += 1
                    else:
                        stats["errors"]    += 1

                if "error" in result:
                    error_buffer.append(result)
                else:
                    successful_buffer.append(result)

                await terminal_log(
                    output_lock,
                    (
                        f"[{datetime.now():%H:%M:%S}] COMPLETED "
                        f"{stats['processed']}/{total} "
                        f"order={result[ID_COL]} "
                        f"classification={result.get('classification', 'ERROR')} "
                        f"latency={latency:.2f}s attempts={attempts}"
                    ),
                )

                # Checkpoint flush
                if len(successful_buffer) >= CHECKPOINT_SIZE:
                    append_rows(SAVE_PATH, successful_buffer)
                    successful_buffer.clear()
                if len(error_buffer) >= CHECKPOINT_SIZE:
                    append_rows(ERROR_PATH, error_buffer)
                    error_buffer.clear()

            finally:
                result_queue.task_done()

        # Shut down
        await producer_task
        await job_queue.join()
        await asyncio.gather(*workers)

    # Final flush
    append_rows(SAVE_PATH, successful_buffer)
    append_rows(ERROR_PATH, error_buffer)

    elapsed = time.time() - start_time
    print(f"\n\n{'═' * 60}")
    print(f"  DONE in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"{'═' * 60}")
    print(f"  ✅  Confirmed : {stats['confirmed']}")
    print(f"  🟡  Possible  : {stats['possible']}")
    print(f"  🔵  Absent    : {stats['absent']}")
    print(f"  ❌  Errors    : {stats['errors']}")
    print(f"  🔄  Retries   : {stats['retries']}")
    print(f"{'═' * 60}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────────
asyncio.run(main())
