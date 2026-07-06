import runpod, time
_IMPORT_TS = time.time()
def handler(job):
    return {"ok": True, "handler_import_ts": round(_IMPORT_TS, 3), "reported_at": round(time.time(), 3)}
runpod.serverless.start({"handler": handler})
