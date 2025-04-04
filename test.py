import requests
import concurrent.futures

# API Endpoint
API_URL = "https://kvzxjmvcpjmqu0-8005.proxy.runpod.net/model_api/"

# File Path (Change this accordingly)
IMAGE_PATH = r"C:\Users\Admin\Downloads\IMG_3267.jpg"

# Number of concurrent requests
NUM_REQUESTS = 10  # Adjust as needed

# Function to send a single request
def send_request(req_count):
    files = {
        "image": (IMAGE_PATH, open(IMAGE_PATH, "rb"), "image/jpeg"),
    }
    data = {"req_count": req_count}

    headers = {
        "accept": "application/json",
    }

    response = requests.post(API_URL, headers=headers, files=files, data=data)
    return response.json()

# Execute requests in parallel
def run_parallel_requests(num_requests):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        results = list(executor.map(send_request, range(1, num_requests + 1)))

    return results

# Run the function
if __name__ == "__main__":
    responses = run_parallel_requests(NUM_REQUESTS)
    for i, response in enumerate(responses):
        print(f"Response {i + 1}: {response}")
