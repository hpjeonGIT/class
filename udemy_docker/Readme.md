## Docker and Kubernetes: The Complete Guide
- Instructor: Stephen Grider

12. Using the docker client
- docker version
- docker run hello-world

13. Container
- When prerequisites of different docker images conflict
    - let each image use different namespace
- namespace vs control groups (cgroups)
    - namespace: isolating resources per process
    - control groups: limit amount of resource used per process

15. Docker run in detail
- docker run <image_name>

16. Overriding default commands
- docker run ImageName Overriding_command
- `docker run busybox ls`
    - Shows the files of the image busybox
- Overriding_command is from the container (image busybox). If the command doesn't exist in the container, the execution will fail

17. Listing running containers
- sudo docker ps
- sudo docker ps --all

18. Container lifecycle
- docker run: docker create + docker start
- docker create: file system snapshot and startup command are injected 
- docker start: 

19. Restarting stopped containers
```bash
hpjeon@hakune:~/hw/class/udemy_docker$ sudo docker run busybox echo hi there
hi there
hpjeon@hakune:~/hw/class/udemy_docker$ sudo docker ps --all
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                     PORTS               NAMES
ac7c1b12ea53        busybox             "echo hi there"     11 seconds ago      Exited (0) 8 seconds ago                       sharp_driscoll
$ sudo docker start ac7c1b12ea53
ac7c1b12ea53  <------------ id only
$ sudo docker start -a ac7c1b12ea53
hi there <--------------- startup command or screen output with -a option
```

20. Removing stopped containers
- docker system prune
    - Removes Stopped containers and all cached items

21. Retrieving log outputs
- docker logs id

22. Stopping containers
- docker stop id: sends SIGTERM. Does some cleaning and takes some time. After 10sec, SIGKILL is sent
- docker kill id: sends SIGKILL. Kills immediately

23. Multi-command containers

24. Executing commands in running containers
- Adding commands into the running containers
- docker exec -it id_container commands_new

25. The purpose of -it flag
- Combination of -i and -t
- -i: interactive
- -t: pretty print

26. Getting a command prompt in a container
- docker exec -it id_container sh
    - Let sh run interactively

27. Starting with a shell
- Run an empty container with shell command

28. Container isolation


29. Creating Docker image
- docker file: contains configuration file
- Specify a base image -> install additional apps -> specify a command to run on container startup

31. Building a docker file
- Dockerfile: Capital D. No extension
```bash
# base image
FROM alpine
# Download and install a dependency
RUN apk add --update redis
# Tell the image what to do @start
CMD ["redis-server"]
```
- Command
```bash
$ sudo docker build .
Sending build context to Docker daemon  2.048kB
Step 1/3 : FROM alpine
latest: Pulling from library/alpine
...
Status: Downloaded newer image for alpine:latest
 ---> 9c842ac49a39
Step 2/3 : RUN apk add --update redis
 ---> Running in 78082be0f329
...
OK: 8 MiB in 15 packages
Removing intermediate container 78082be0f329
 ---> 177a92a24492
Step 3/3 : CMD ["redis-server"]
 ---> Running in 81434ab1f7cd
Removing intermediate container 81434ab1f7cd
 ---> 3c6f23e779f8
Successfully built 3c6f23e779f8
$ sudo docker run 3c6f23e779f8
# Now redis runs
```

33. What's a base image?

34. The build processs in detail

35. A brief recap
- FROM: alphine => Alpine image
- RUN: apk add --update redis => The first temporary image with redis
- CMD: ["redis-server"] => Another temporary image with startup command

36. Rebuilds with cache
- When the base image/apps are in the cache, docker will recycle them, accelerating building images

37. Tagging an image
- `docker build -t my_docker_id/proj_name:version .`
