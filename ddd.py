@router.get("/")
def health():
    "Basic health check for the service."
    return {"status": "OK"}
