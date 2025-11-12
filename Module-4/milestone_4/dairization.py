import time
import requests
import json


def get_diarization_result(job_id, api_key, poll_interval=10, max_checks=60):
    """
    Poll the pyannote jobs endpoint until the job finishes.

    Returns the diarization output (usually a list) on success, or None on failure.
    Adds more verbose logs for debugging HTTP and JSON issues.
    """
    if not job_id:
        print("❌ No job_id provided to get_diarization_result.")
        return None

    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        for attempt in range(1, max_checks + 1):
            try:
                response = requests.get(f"https://api.pyannote.ai/v1/jobs/{job_id}", headers=headers, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error while checking job status (attempt {attempt}): {e}")
                time.sleep(poll_interval)
                continue

            # log HTTP-level issues
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code} when polling job {job_id}: {response.text}")
                # don't break immediately; some APIs return 202/204 during processing
                if attempt >= max_checks:
                    return None
                time.sleep(poll_interval)
                continue

            # parse JSON
            try:
                data = response.json()
            except ValueError:
                print(f"❌ Failed to parse JSON (attempt {attempt}). Response text:\n{response.text}")
                if attempt >= max_checks:
                    return None
                time.sleep(poll_interval)
                continue

            # debug dump of keys for troubleshooting
            status = data.get("status")
            print(f"🔎 Job {job_id} status check #{attempt}: status={status}")

            if status in ["succeeded", "failed", "canceled"]:
                if status == "succeeded":
                    output = data.get("output")
                    diarization = None
                    if isinstance(output, dict):
                        diarization = output.get("diarization") or output.get("segments") or output.get("result")
                    else:
                        diarization = data.get("output")

                    if diarization is None:
                        print("⚠️ Job succeeded but no 'diarization' field found in output. Dumping output for inspection:")
                        print(json.dumps(data.get("output", {}), indent=2, ensure_ascii=False))
                        return None

                    print("✅ Job completed successfully and diarization extracted.")
                    return diarization
                else:
                    print(f"⚠️ Job finished with status: {status}")
                    return None

            # still running
            print(f"⏳ Job status: {status} — waiting {poll_interval}s (check {attempt}/{max_checks})...")
            time.sleep(poll_interval)

        print(f"❌ Exceeded max checks ({max_checks}) while polling job {job_id}.")
        return None

    except Exception as e:
        print(f"⚠️ Unexpected error occurred while polling job {job_id}: {e}")
        return None


if __name__ == "__main__":
    job_id = "9361833d-33d2-48bd-b227-2c4cf861db1b"  
    get_diarization_result(job_id)