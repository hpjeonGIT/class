## Kubernetes for the Absolute Beginners - Hands-on
- Instructor: Mumshad Mannambeth

## Section 1: Introduction

1. Introduction

2. The Kubernetes Course Series

3. Coures Resources

4. Course Release Notes

## Section 2: Kubernetes Overview

5. Containers Overview
- Kubernetes or K8s
  - Container + Orchestration
- Containers
  - Docker uses lxc
    - Shares the underlying kernel
  - An isolated environment
  - Not virtualize like hypervisor
- Container vs image
  - An image runs on a container form

6. Container Orchestration
- Mesos by Apache
- Kubernetes by Google
- Docker Swarm by Docker

7. Kubernetes Architecture
- Nodes: machine. VM. Containers will be launched by Kubernetes
- Cluster: set of nodes
- Master: watches over nodes
- Kubernetes components
  - API server
  - etcd: keyvalue store
  - kubelet: agents running on each cluster
  - Container runtime: docker, rkt, cri-o, ...
  - Controller:
  - Scheduler
- Master vs worker nodes
- kubectl
  - kubectl run hello-minikube
  - kubectl cluster-info
  - kubectl get nodes

8. Student Perferences

## Section 3: Setup Kubernetes

9. Kubernetes Setup - Introduction Minikube
- Kubernetes tools
  - Minikube
  - MiciroK8s
  - Kubeadm
- Matster node
  - kube-apiserver
  - etcd
  - node-controller
  - Rreplica-controller
- Worker node
  - kubelet
  - Container Runtime (docker)
- Minikube
  - All of Master and Worker nodes

10. Reference

11. Demo - Minikube
- How to check if the virtualization is supported in the pc
  - `grep vm /proc/cpuinfo`
- minikube start --driver=virtualbox # for virtulbox
- minikube start
  - Takes a few minutes
- minikube status
- First tutorial
  - https://kubernetes.io/docs/tutorials/hello-minikube/
```bash
$ minikube start
😄  minikube v1.25.2 on Ubuntu 18.04
✨  Using the docker driver based on existing profile
👍  Starting control plane node minikube in cluster minikube
🚜  Pulling base image ...
🔄  Restarting existing docker container for "minikube" ...
🐳  Preparing Kubernetes v1.23.3 on Docker 20.10.12 ...
    ▪ kubelet.housekeeping-interval=5m
🔎  Verifying Kubernetes components...
    ▪ Using image gcr.io/k8s-minikube/storage-provisioner:v5
🌟  Enabled addons: storage-provisioner, default-storageclass
🏄  Done! kubectl is now configured to use "minikube" cluster and "default" namespace by default
$ kubectl version
...
$ minikube status
...
$ kubectl get nodes
...
$ kubectl create deployment hello-node --image=k8s.gcr.io/echoserver:1.4
...
$ kubectl get deployments
...
$ kubectl get pods
...
$ kubectl get events
...
$ kubectl expose deployment hello-node --type=LoadBalancer --port=8080
...
$ minikube service hello-node --url http://192.168.99.100:31391 
http://192.168.49.2:30889 ## open this at a web-browser
$ kubectl get services
NAME               TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)          AGE
client-node-port   NodePort       10.110.103.102   <none>        3050:31515/TCP   55d
hello-node         LoadBalancer   10.110.242.218   <pending>     8080:32041/TCP   6s
kubernetes         ClusterIP      10.96.0.1        <none>        443/TCP          55d
$ kubectl delete service hello-node
service "hello-node" deleted
$ kubectl delete deployment hello-node
deployment.apps "hello-node" deleted
$ minikube stop
```

## Section 4: Kubernetes Concepts

12. PODs
- Pod to container is 1:1
- A single POD can have multiple containers
- How to deploy a POD
  - kubectl run nginx --image nginx
    - Images from Docker hub
- kubectl get pods
  - lists the available pods  

13. Demo - PODS
```bash
$ minikube start
$ kubectl run nginx --image=nginx
$ kubectl get pods
$ kubectl describe pod nginx
$ kubectl get pods -o wide
```

14. Reference - PODs

## Sectin 5: YAML Introduction

15. Introduction to YAML
- Key value pair
```yaml
A: Apple
B: Banana
```
  - After colon, one space is necessary
- Array/List : ordered
```yaml
Fruits:
- Orange
- Apple
```
  - Dash means elements
```yaml
Payslips:
    - Month: June
      Wage: 4000
    - Month: July
      Wage: 4500
```
  - In this example, Monthl and Wage are included in a singe list item. Must have same distance from the first column
- Dictonary/Map: unordered
```
Banana:
    Calories: 105
    Fat: 0.4 g
```
  - Must have the same size of space in the elements
- Mixture of key value/Dictionary/Lists
```
Fruits:
- Banana:
    Calories: 105
    Fat: 0.4 g
- Grape:
    Calories: 62
    Fat: 0.2 g
```
- Hash(#) is the comment line

16. Introduction to Coding Exercise
- To check if a yaml is valid, use: http://www.yamllint.com/

17. Coding Exercises - Answer keys
- https://github.com/mmumshad/kubernetes-training-answers
```yaml
Employee:
  Name: Jacob
  Sex: Male
  Age: 30
  Title: Systems Engineer
  Projects:
    - Automation
    - Support
  Payslips:
    - Month: June
      Wage: 4000
    - Month: July
      Wage: 4500
    - Month: August
      Wage: 4000
```

## Section 6: Kubernetes Concepts - PODS, ReplicaSets, Deployments

18. PODs with YAMLd
- pod-definition.yml
  - apiVersion: string
  - kind: string
    - Pod, Service, ReplicaSet, Deployment
    - Case-sensitive
  - metaData: dictionary
    - name
    - labels
      - app, type
  - spec: dictionary
    - containers: List/Array
      - name, image
- Commands
  - kubectl get pods
  - kubectl describe pod myapp-pod

19. Demo - PODs with YAML
- pod.yaml
```yaml
apiVersion: v1
kind: Pod
metadata: 
  name: nginx
  labels: 
    app: nginx
    tier: frontend
spec:
  containers:
  - name: nginx
    image: nginx
```
- kubectl apply -f pod.yaml
- kubectl get pods

20. Tips & Tricks - Developing Kubenetes Manifest files with VS code
- Passing environmental variables
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: postgres
  labels:
    tier: db-tier
spec:
  containers:
    - name: postgres
      image: postgres
      env: 
        - name: POSTGRES_PASSWORD
          value: mysecretpassword
```

21. Demo - How to Access the Labs?

22. Accessing the Labs
- https://uklabs.kodekloud.com/courses/labs-kubernetes-for-the-absolute-beginners-hands-on/
- A coupon is shown in the lecture note

23. Hands-On Labs - Familiarize with the lab environment

24. Hands-On Labs
- kubectl get nodes : number of nodes running
- kubectl version : version
- kubectl get nodes -o wide: OS info of the nodes
- kubectl get pods: number of pods running
  - READY column shows the running containers/total containers
- kubectl run nginx --image=nginx
  - This image yielded 4 pods
- kubectl descibe pod nginx: shows what image is used to create the pod
- kubectl get pods -o wide: shows which node those pods are running on 
- kubectl delete pod webapp: deletes webapp pod
- kubectl run redis --image=redis123
- kubectl edit pod redis
  - vi editor opens
  - Exit after saving. Then automatically restarts

25. Solution: Pods with YAML lab

26. Replication Controllers and ReplicaSets
- Multiple pods or a single pod with replication controller for high availability
- Replication controller vs replica set
  - We are migrating into replica set
- Replication controller
  - rc-definition.yml
```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: myapp-rc
  labels:
    app: myapp
    type: front-end
spec:
  template: 
    metadata:
      name: myapp-pod
      labels:
        app: myapp
        type: frontend
    spec:
      containers:
      - name: nginx-container
        image: nginx
  replicas: 3
```
  - kubectl create -f rc-definition.yml
  - kubectl get replicationcontroller
  - kubectl get pods
- Replica set
  - replicaset-definition.yml
```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: myapp-replicaset
  labels:
    app: myapp
    type: front-end
spec:
  template: 
    metadata:
      name: myapp-pod
      labels:
        app: myapp
        type: front-end
    spec:
      containers:
      - name: nginx-container
        image: nginx
  replicas: 3
  selector: 
    matchLabels:
      type: front-end
```
  - Needs selector key than Replica-Controller
    - In matchLabels, any keyword from labes: can be taken, such as type, app, ...
  - kubectl create -f replicaset-definition.yml
  - kubectl get replicaset
  - kubectl get pods
  - How to scale:
    - Update the yaml and reload
      - kubectl replace -f replicaset-definiton.yml
    - Or just update using options
      - kubectl scale --replicas=6 -f replicaset-defintion.yml
      - kubectl scale --replicas=6 replicaset myapp-replicaset
      - The file of replicaset-definition.yml is not updated though
  - To delete all underyling PODs
    - kubectl delete replicaset myapp-replicaset
- If a new POD has the same type of replicaset, it will be terminated by Replica controller
- kubectl edit replicaset myapp-replicaset
  - Edit will pop up
```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: frontend
  labels:
    app: mywebsite
    tier: frontend
spec:
  replicas: 4
  template:
    metadata:
      name: myapp-pod
      labels:
        app: myapp
    spec:
      containers:
        - name: nginx
          image: nginx
  selector:
    matchLabels:
      app: myapp
```

27. Demo - ReplicaSets

28. Hands-On Labs
- kubectl get pods: finds the number of pods
- kubectl get replicaset: finds the number of replicaset
- kubectl get pods: shows what image has been used
- kubectl delete pod new-abc-123: delete a pod new-abc-123
- kubectl delete replicaset replicaset-1: delete the entire pods of replicaset-1
- kubectl edit relicaset new-replica-set: edit the new-replica-set on-the-fly
  - Need to delete existing pods. Then new pods will be produced with the update
  - When increasing the number of replicas, it is done automatically.
  - When reducing the number of replicas, it is done automatically but takes time
- kubectl scale --replicas=6 replicaset myapp-replicaset

29. Solution - ReplicaSets

30. Deployments
- How to deploy
- How to roll-back
- Deployment
  - Definition
  - deployment-definition.yml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  labels:
    app: mywebsite
    tier: frontend
spec:
  replicas: 4
  template:
    metadata:
      name: myapp-pod
      labels:
        app: myapp
    spec:
      containers:
        - name: nginx
          image: nginx
  selector:
    matchLabels:
      app: myapp
```
  - Basically same to Replicaset except the kind: key
  - kubectl create -f deployment-definition.yml
  - kubectl get deployments
  - kubectl get replicaset
    - Note that replicaset is shown with the deployment
  - kubectl get pods
  - kubectl get all
    - Shows deployment, replicaset, pods

31. Demo - Deployments
- deployment/deployment.yml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-deployment
  labels:
    app: nginx
    tier: frontend
spec:
  replicas: 3
  template:
    metadata:
      name: nginx-2
      labels:
        app: myapp
    spec:
      containers:
        - name: nginx
          image: nginx
  selector:
    matchLabels:
      app: myapp
```
- kubectl create -f deployment/deployment.yml
- kubectl describe deployment myapp-deployment

32. Hands-On Labs

33. Solution - Deployments

34. Deployments - Update and Rollback
- Rollout command
```bash
$ kubectl rollout status  deployment/myapp-deployment
deployment "myapp-deployment" successfully rolled out
$ kubectl rollout history deployment/myapp-deployment
deployment.apps/myapp-deployment 
REVISION  CHANGE-CAUSE
1         <none>
```
- Deployment Strategy
  - Recreate: down all pods then up new pods. Down time exists
  - Rolling update: down/up pods one by one
- kubectl apply -f deployment-definition.yml
  - New rollout is applied
- kubectl set image deployment/myapp-deployment nginx=nginx:1.9.1
  - Definition file is not changed
```bash
$ kubectl describe deployment myapp-deployment
Name:                   myapp-deployment
Namespace:              default
CreationTimestamp:      Fri, 10 Jun 2022 13:53:44 -0400
Labels:                 app=nginx
                        tier=frontend
Annotations:            deployment.kubernetes.io/revision: 1
Selector:               app=myapp
Replicas:               3 desired | 3 updated | 3 total | 3 available | 0 unavailable
StrategyType:           RollingUpdate
```
  - Default is Rolling Update
- Upgrades
```bash
$ kubectl apply -f deployment-definition.yml
or
$ kubectl set image deployment/myapp-deployment nginx=nginx:1.9.1
# or update using kubectl edit deployment myapp-deployment. When exit, update is applied automatically
$ kubectl get replicaset
NAME                          DESIRED   CURRENT   READY   AGE
myapp-deployment-5947878645   3         3         2       24s
myapp-deployment-6ffd468748   1         1         1       39m
```
  - List of relicaset is changed over time
  - Old replicaset is not deleted for the rollback
- Rollback
  - kubectl rollout undo deployment/myapp-deployment

35. Demo - Deployments - Update and Rollback

36. Lab: Practice Testing Rolling Updates and Rollbacks
- How many pods can be down at the same time?
  - Check the strategy from kubectl edit deployment mydeployment
```yaml
  strategy:
    rollingupdate:
      maxSurge: 25%
      maxUnavailable: 25%
```

## Section 7: Networking in Kubernetes

37. Basics of Networking in Kubernetes
- All pods are connected to internal address 10.244.0.0
- Cluster networking
  - PODs in a single node may have same IP of another nodes
  - Kubernetes doesn't configure different IPs over those PODs over different nodes
  - Admins must configure them differently so all PODs can communicate each other
    - Flannel, cilium, NSX ...

## Section 8: Services

38. Services - NodePort
- How an external user can access the service in a pod?
  - client: 192.168.1.10
  - node: 192.168.1.2
  - POD: 10.244.0.2
    - How to map POD to node IP?
    - Node porting service
- Service Types
  - NodePort
    - Range of 30000-32767
    - service-definition.yml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  type: NodePort
  ports:
  - targetPort: 80
    port: 80
    nodePort: 30008
  selector:
    app: myapp
    type: front-end
```
    - targetPort is the port number of the pod
    - port is the port number of service
    - nodePort is the port number of a node where the POD lives
    - The section of selector: is from the pod-definition.yml or deployment-definitio.yml, mapping the POD into the service
    - kubectl create -f service-defintion.yml
    - kubectl get services
  - ClusterIP
  - LoadBalancer

39. Demo - Services
- service/service-definition.yaml
```
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  type: NodePort
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30004
  selector:
    app: myapp
```
- kubectl create -f service/service-definition.yaml
- kubectl get service
- minikube service myapp-service --url 
  - Shows the IP to connect

40. Services - ClusterIP
- service-definition.yml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: back-end
spec:
  type: ClusterIP
  ports:
  - targetPort: 80
    port: 80
  selector:
    app: myapp
    type: back-end
```
- kubectl create -f service/service-definition.yaml
- kubectl get service

41. Services - Load Balancer

42. Hands-On Labs

43. Solution - Services
- Making service definition yaml
  - kubectl expose deployment simple-webapp-deployment --name=webapp-service --target-port=8080 --type-NodePort --port=8080 --dry-run=client -o yaml > svc.yaml

## Section 9: Microservices Architecture

44. Microservices Application
- Sample application - voting application
  - voting-app (python) -> in-memory DB (redis) -> PostgreSQL db -> result-app (nodejs)
- Running on a host server
  - docker run -d --name=redis redis
  - docker run -d --name=db postgres:9.4
  - docker run -d --name=vote -p 5000:80 --link redis:redis voting-app
  - docker run -d --name=result -p 5001:80 --link db:db result-app
  - docker run -d --name=worker --link db:db --link redis:redis worker

45. Microservices Application on Kubernetes
- Repeat the above deployment on Kubernetes
- Deploy PODs
- Create services (ClusterIP)
  - redis for python/redis
  - db for result vs postgres
- Create services (NodePort)
  - voting-app
  - result-app

46. Demo - Deploying Microservices Application on Kubernetes
- defintion files
```yaml
==> postgres-pod.yaml <==
apiVersion: v1
kind: Pod
metadata:
  name: postgres-pod
  labels:
    name: postgres-pod
    app: demo-voting-app
spec:
  containers:
  - name: postgres
    image: postgres
    ports:
      - containerPort: 5432
    env:
      - name: POSTGRES_USER
        value: "postgres"
      - name: POSTGRES_PASSWORD
        value: "postgres"
==> postgres-service.yaml <==
apiVersion: v1
kind: Service
metadata:
  name: db
  labels:
    name: db-service
    app: demo-voting-app
spec:
  ports:
    - port: 5432
      targetPort: 5432
  selector:
    name: postgres-pod
    app: demo-voting-app
==> redis-pod.yaml <==
apiVersion: v1
kind: Pod
metadata:
  name: redis-pod
  labels:
    name: redis-pod
    app: demo-voting-app
spec:
  containers:
  - name: redis
    image: redis
    ports:
      - containerPort: 6379
==> redis-service.yaml <==
apiVersion: v1
kind: Service
metadata:
  name: redis
  labels:
    name: redis-service
    app: demo-voting-app
spec:
  ports:
    - port: 6379
      targetPort: 6379
  selector:
    name: redis-pod
    app: demo-voting-app
==> result-app-pod.yaml <==
apiVersion: v1
kind: Pod
metadata:
  name: result-app-pod
  labels:
    name: result-app-pod
    app: demo-voting-app
spec:
  containers:
  - name: result-app
    image: kodekloud/examplevotingapp_result:v1
    ports:
      - containerPort: 80
==> result-app-service.yaml <==
apiVersion: v1
kind: Service
metadata:
  name: result-service
  labels:
    name: result-service
    app: demo-voting-app
spec:
  type: NodePort
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30005
  selector:
    name: result-app-pod
    app: demo-voting-app
==> voting-app-pod.yaml <==
apiVersion: v1
kind: Pod
metadata:
  name: voting-app-pod
  labels:
    name: voting-app-pod
    app: demo-voting-app
spec:
  containers:
  - name: voting-app
    image: kodekloud/examplevotingapp_vote:v1
    ports:
      - containerPort: 80
==> voting-app-service.yaml <==
apiVersion: v1
kind: Service
metadata:
  name: voting-service
  labels:
    name: voting-service
    app: demo-voting-app
spec:
  type: NodePort
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30004
  selector:
    name: voting-app-pod
    app: demo-voting-app
==> worker-app-pod.yaml <==
apiVersion: v1
kind: Pod
metadata:
  name: worker-app-pod
  labels:
    name: worker-app-pod
    app: demo-voting-app
spec:
  containers:
  - name: worker-app
    image: kodekloud/examplevotingapp_worker:v1
```
- Commands
```bash
kubectl create -f voting-app-pod.yaml 
kubectl create -f voting-app-service.yaml 
kubectl create -f redis-pod.yaml 
kubectl create -f redis-service.yaml 
kubectl create -f postgres-pod.yaml 
kubectl create -f postgres-service.yaml 
kubectl create -f worker-app-pod.yaml 
kubectl create -f result-app-pod.yaml 
kubectl create -f result-app-service.yaml 
kubectl get pods,svc
minikube service voting-service --url
minikube service result-service --url
http://192.168.49.2:30005 # open this page in a web browser
```
- Issues
```
pod/worker-app-pod   0/1     CrashLoopBackOff   5 (2m36s ago)   11m
```
  - Run dry-run and get yaml configuration
```
$ kubectl create deployment test --image=kodekloud/examplevotingapp_worker --dry-run=client -oyaml
apiVersion: apps/v1
kind: Deployment
metadata:
  creationTimestamp: null
  labels:
    app: test
  name: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test
  strategy: {}
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: test
    spec:
      containers:
      - image: kodekloud/examplevotingapp_worker
        name: examplevotingapp-worker-nlrzk
        resources: {}
status: {}
```
  - All source: https://github.com/mmumshad/example-voting-app-kubernetes-v2
  - https://www.udemy.com/course/learn-kubernetes/learn/lecture/21126612#questions/12458624

47. Demo - Deploying Microservices Application on Kubernetes with Deployments

## Section 10: Kubernetes on Cloud

48. Kubernetes on Cloud - Introduction
- self hosted/turnkey solutions
  - You provision VMs
  - You configure VMs
  - You use scripts to deploy cluster
  - You maintain VMs yourself
  - eg: Kubernetes on AWS using kops or Kubeone
- Hosted solutions
  - Kubernetes-As-A-Service
  - Provider provisions VMs
  - Provider installs Kubernetes
  - Provide maintains VMs
  - eg: Google Container Engine (GKE), Azure Kubernetes Service (AKS), Amazon Elastic Kubernetes Service (EKS)

49. Kubernetes on GCP (GKE)

50. Reference - Google Cloud
- Free tier:  https://cloud.google.com/free/ 

51. Kubernetes on AWS (EKS)

52. Kubernetes on Azure (AKS)

## Section 11: Conclusion

53. Conclusion

## Section 12: Appendix - Setup Multi Node Cluster using Kubeadm

54. Reference

55. Kubernetes Setup - Kubeadm
- For multi-node clusters
- Steps
  - Provision VMs
  - Install docker on each node
  - Install kubeadm on each node
  - Initialize master node
  - Configure POD network

56. Demo - Setup Lab - VirtualBox
- Requires VirtualBox and Vagrant
- https://github.com/kodekloudhub/certified-kubernetes-administrator-course

57. Demo - Provision cluster usign Kubeadmin

58. Bonus Lecture: Kubernetes Series of Courses

Q: Do we need kubernetes for HPC?
