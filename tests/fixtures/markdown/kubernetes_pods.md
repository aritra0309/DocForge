# Pods

*Pods* are the smallest deployable units of computing that you can create and manage in Kubernetes.

A Pod (as in a pod of whales or pea pod) is a group of one or more containers, with shared storage and network resources, and a specification for how to run the containers.

> **Note:** You need to install a container runtime into each node in the cluster so that Pods can run there.

## Pod templates

The sample below is a manifest for a simple Job with a template that starts one container.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hello
spec:
  template:
    spec:
      containers:
      - name: hello
        image: busybox:1.28
        command: ['sh', '-c', 'echo "Hello, Kubernetes!" && sleep 3600']
      restartPolicy: OnFailure
```

## Pod Lifecycle

Pods follow a defined lifecycle, starting in the `Pending` phase, moving through `Running` if at least one of its primary containers starts OK, and then through either the `Succeeded` or `Failed` phases depending on whether any container in the Pod terminated in failure.

| Value | Description |
| --- | --- |
| Pending | Pod accepted but not started |
| Running | Pod bound to a node, at least one container is running |
| Succeeded | All containers terminated successfully |
| Failed | All containers terminated, at least one failed |

See [Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle.html) for more details.
