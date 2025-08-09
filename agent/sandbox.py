import docker
import os

DOCKER_IMAGE_NAME = "data-analyst-sandbox"

def create_sandbox_image():
    client = docker.from_env()
    try:
        client.images.get(DOCKER_IMAGE_NAME)
        print(f"Image '{DOCKER_IMAGE_NAME}' exists, skipping build.")
    except docker.errors.ImageNotFound:
        print(f"Building image '{DOCKER_IMAGE_NAME}'...")
        try:
            client.images.build(path=".", tag=DOCKER_IMAGE_NAME, rm=True)
            print(f"Image '{DOCKER_IMAGE_NAME}' built successfully.")
        except docker.errors.BuildError as e:
            print(f"Build error: {e}")
            for line in e.build_log:
                if 'stream' in line:
                    print(line['stream'].strip())
            raise

def run_in_sandbox(workspace_dir: str, code: str) -> tuple[str, str]:
    client = docker.from_env()
    full_path = os.path.abspath(workspace_dir)
    container = None

    MEM_LIMIT = os.getenv("SANDBOX_MEM_LIMIT", "512m")
    CPU_PERIOD = int(os.getenv("SANDBOX_CPU_PERIOD", "100000"))
    CPU_QUOTA = int(os.getenv("SANDBOX_CPU_QUOTA", "50000"))

    try:
        container = client.containers.run(
            image=DOCKER_IMAGE_NAME,
            command=["python", "-c", code],
            volumes={full_path: {'bind': '/workspace', 'mode': 'rw'}},
            working_dir="/workspace",
            detach=True,
            network_mode="none",
            mem_limit=MEM_LIMIT,
            cpu_period=CPU_PERIOD,
            cpu_quota=CPU_QUOTA,
        )
        try:
            container.wait(timeout=120)
        except docker.errors.APIError as e:
            print(f"Timeout/API error while waiting: {e}")
            container.kill()
            return "", f"Timeout or execution error: {e}"

        try:
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error fetching stdout logs: {e}")
            stdout = ""

        try:
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error fetching stderr logs: {e}")
            stderr = ""

        return stdout, stderr

    except docker.errors.ContainerError as e:
        print(f"Container execution error: {e}")
        return "", str(e)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "", str(e)
    finally:
        if container:
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
