# Batch Progress

The server now shows live batch progress in the terminal while `/v1/batches` jobs are executing.

- Interactive terminals get a `tqdm` progress bar per active batch job.
- Concurrent batch jobs get separate progress-bar positions so their output does not overwrite each other.
- Each processed item advances the bar once, whether the item succeeded or failed.
- Non-interactive environments do not get animated progress bars. They receive start and finish log lines instead.
- Batch progress is internal to the server process only. The HTTP batch API remains unchanged.
