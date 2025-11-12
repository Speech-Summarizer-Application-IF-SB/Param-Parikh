import requests
import json

def get_job_id(input_path, api_key):
    try:
        # === Configuration ===

        object_key = "myMeeting"        # Unique identifier for your file

        if not api_key:
            raise ValueError("Missing API key. Set PYANNOTE_API_KEY as an environment variable.")

        # === Step 1: Request a pre-signed PUT URL ===
        try:
            response = requests.post(
                "https://api.pyannote.ai/v1/media/input",
                json={"url": f"media://{object_key}"},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30,
            )
            if response.status_code != 200:
                print(f"❌ Pre-signed URL request returned HTTP {response.status_code}: {response.text}")
                response.raise_for_status()
            data = response.json()
            presigned_url = data.get("url") or data.get("presignedUrl") or data.get("presigned_url")
            if not presigned_url:
                print("❌ Could not find presigned URL in response. Dumping response:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                raise ValueError("Missing presigned URL in media/input response")
            print(f"✅ Pre-signed URL obtained for object key '{object_key}'")
        except requests.exceptions.RequestException as e:
            print("❌ Error requesting pre-signed URL:", e)
            raise

        # === Step 2: Upload your audio file ===
        try:
            print(f"⬆️  Uploading {input_path} to {presigned_url} ...")
            with open(input_path, "rb") as input_file:
                upload_response = requests.put(presigned_url, data=input_file, timeout=60)
                if upload_response.status_code not in (200, 201, 204):
                    print(f"❌ Upload returned HTTP {upload_response.status_code}: {upload_response.text}")
                    upload_response.raise_for_status()
            print("✅ File uploaded successfully!")
        except FileNotFoundError:
            print(f"❌ File not found: {input_path}")
            raise
        except requests.exceptions.RequestException as e:
            print("❌ Error uploading file:", e)
            raise

        # === Step 3 : Start diarization ===
        try:
            print("🚀 Starting diarization job...")
            diarize_response = requests.post(
                "https://api.pyannote.ai/v1/diarize",
                json={"url": f"media://{object_key}"},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30,
            )

            if diarize_response.status_code not in (200, 201, 202):
                print(f"❌ Diarize request returned HTTP {diarize_response.status_code}: {diarize_response.text}")
                diarize_response.raise_for_status()

            resp_json = diarize_response.json()
            print("✅ Diarization job created successfully! Response:")
            print(json.dumps(resp_json, indent=2, ensure_ascii=False))

            # Job ID key can vary by API; try a few possibilities
            job_id = resp_json.get("jobId") or resp_json.get("job_id") or resp_json.get("id")
            if not job_id:
                raise ValueError("No job id found in diarize response")

            return job_id
        except requests.exceptions.RequestException as e:
            print("❌ Error starting diarization job:", e)
            raise

    except Exception as e:
        print("⚠️ Unexpected error occurred:", e)