# Deployments

A *Deployment* provides declarative updates for Pods and ReplicaSets.

You describe a *desired state* in a Deployment, and the Deployment Controller changes the actual state to the desired state at a controlled rate.

## Creating a Deployment

The following is an example of a Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
```

To create this Deployment, run:

```bash
kubectl apply -f https://k8s.io/examples/controllers/nginx-deployment.yaml
```

## Updating a Deployment

```bash
kubectl set image deployment/nginx-deployment nginx=nginx:1.16.1
```

## Rolling Back a Deployment

```bash
kubectl rollout undo deployment/nginx-deployment
```

## Scaling a Deployment

```bash
kubectl scale deployment/nginx-deployment --replicas=10
```
