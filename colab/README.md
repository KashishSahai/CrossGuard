# Google Colab CUDA Worker for CrossGuard

This directory contains the deployment infrastructure for running CrossGuard YOLOv8n object detection on a remote NVIDIA GPU (e.g. Tesla T4 / A100) hosted on Google Colab.

> **Important Compute Notice**:
> Google Colab is an ephemeral compute environment. The worker URL may change when the runtime restarts, but CrossGuard discovers the newly registered URL automatically via the GitHub Gist registry without requiring manual edits to `.env`.
> Google Colab is designed as a development and research execution environment, not a permanent 24/7 production server.

---

## Complete 10-Step Deployment Process

1. **Open CrossGuard GPU Worker Colab notebook**:
   Open [`colab/CrossGuard_GPU_Worker.ipynb`](CrossGuard_GPU_Worker.ipynb) in [Google Colab](https://colab.research.google.com).

2. **Select NVIDIA T4 GPU runtime**:
   Navigate to **Runtime** → **Change runtime type**, select **T4 GPU** (or any available GPU accelerator), and click **Save**.

3. **Configure Colab Secrets (Optional, Recommended for Auto-Discovery)**:
   In the left sidebar of Colab, click the **Secrets** (key) icon and add:
   - `CROSSGUARD_GPU_REGISTRY_GIST_ID`: Your GitHub Gist ID (e.g. from `https://gist.github.com`).
   - `CROSSGUARD_GPU_REGISTRY_GITHUB_TOKEN`: A GitHub Personal Access Token with `gist` scope.

4. **Run the worker**:
   Execute all cells (**Runtime** → **Run all**).

5. **Worker creates Cloudflare tunnel**:
   The notebook initializes the FastAPI server and starts a Cloudflare tunnel (`https://<subdomain>.trycloudflare.com`).

6. **Worker automatically registers the current tunnel URL**:
   The worker publishes its new tunnel URL, GPU model (`Tesla T4`), and timestamp to the fixed GitHub Gist, starting a background heartbeat every 60 seconds.
   The notebook displays:
   ```text
   ======================================================================
   CrossGuard GPU Worker
   GPU: Tesla T4
   CUDA: available
   Worker URL: https://example.trycloudflare.com
   Registry: updated
   Status: ONLINE
   ======================================================================
   ```

7. **Open CrossGuard**:
   Launch your local CrossGuard application:
   ```bash
   streamlit run app.py
   ```
   Ensure `CROSSGUARD_GPU_REGISTRY_GIST_ID` is set in your `.env` file.

8. **Select Auto or GPU (CUDA)**:
   In CrossGuard, set **Processing Device** to **GPU (CUDA)** or **Auto**.

9. **CrossGuard discovers the current worker URL & verifies Tesla T4 status**:
   CrossGuard reads the fixed Gist, detects the active Colab worker URL, runs a health check, and displays:
   ```text
   GPU Status: Connected • WORKER REACHABLE + CUDA AVAILABLE • GPU: Tesla T4 • Worker Source: Registry
   ```

10. **Process the video**:
    Click **Step 1 Frame** or **Run Sequence** in **Live Vision**. CrossGuard offloads YOLOv8n inference to the remote Tesla T4 while retaining tracking, homography, spatial analysis, FSM, and Bloom City locally.

---

## Manual Fallback (Without GitHub Gist)

If you prefer not to use the dynamic Gist registry:
1. Run the Colab notebook.
2. Copy the printed `Worker URL: https://...trycloudflare.com`.
3. Set in your `.env` file:
   ```env
   CROSSGUARD_GPU_WORKER_URL=https://your-tunnel-url.trycloudflare.com
   ```
4. CrossGuard will use this static fallback URL directly.
