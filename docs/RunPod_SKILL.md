# RunPod MCP Skill Guide

This document explains how to use the RunPod MCP (Model Context Protocol) to manage pods programmatically.

## Available Tools

The runpod MCP provides several tools to manage your pods:

### 1. List Pods
Use `mcp_runpod_list-pods` to list all your pods and their current statuses (e.g., RUNNING, EXITED).
```json
{
  "tool": "mcp_runpod_list-pods",
  "parameters": {}
}
```

### 2. Get Pod Details
Use `mcp_runpod_get-pod` to retrieve detailed information for a specific pod, including the mapped SSH ports, public IP, and specs.
```json
{
  "tool": "mcp_runpod_get-pod",
  "parameters": {
    "podId": "<POD_ID>",
    "includeMachine": true
  }
}
```

### 3. Start a Pod
Use `mcp_runpod_start-pod` to boot up a pod that is currently in the EXITED state.
```json
{
  "tool": "mcp_runpod_start-pod",
  "parameters": {
    "podId": "<POD_ID>"
  }
}
```

### 4. Stop a Pod
Use `mcp_runpod_stop-pod` to stop a running pod (putting it in the EXITED state). This is useful to save costs when not actively running a pipeline.
```json
{
  "tool": "mcp_runpod_stop-pod",
  "parameters": {
    "podId": "<POD_ID>"
  }
}
```

### 5. Create a New Pod
Use `mcp_runpod_create-pod` to provision a new pod. You must provide a valid Docker image and configuration parameters. Use key ~/.ssh/id_ed25519.pub as the public key.

```json
{
  "tool": "mcp_runpod_create-pod",
  "parameters": {
    "cloudType": "SECURE",
    "gpuCount": 1,
    "volumeInGb": 80,
    "containerDiskInGb": 50,
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "name": "mindeye-new-pod",
    "ports": ["22/tcp"],
    "volumeMountPath": "/workspace",
    "env": {
      "PUBLIC_KEY": "ssh-ed25519 AAA... user@example.com"
    }
  }
}
```

### 6. Delete a Pod
Use `mcp_runpod_delete-pod` to permanently terminate a pod.
```json
{
  "tool": "mcp_runpod_delete-pod",
  "parameters": {
    "podId": "<POD_ID>"
  }
}
```
