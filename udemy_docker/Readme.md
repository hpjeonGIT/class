## Docker and Kubernetes: The Complete Guide
- Instructor: Stephen Grider

11. Installing docker on Linux
- Ref: https://docs.docker.com/engine/install/ubuntu/
```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt autoremove
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo   "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo docker run hello-world
```

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
- Specify a base image: FROM alpine
- Run some commands to install additional programs: RUN apk add --update redis
- Specify a command to run on container startup: CMD["redis-server"]

35. A brief recap
- FROM: alphine => Alpine image
- RUN: apk add --update redis => The first temporary image with redis
- CMD: ["redis-server"] => Another temporary image with startup command

36. Rebuilds with cache
- When the base image/apps are in the cache, docker will recycle them, accelerating building images

37. Tagging an image
- `docker build -t my_docker_id/proj_name:version .`

40. Project outline
- Create nodeJS web app
- Create a dockerfile
- Build image from docker file
- Run image as container
- Connect to web app from a browser

41. Node server setup
- mkdir simpleWeb
- cd simpleWeb
- Prepare package.json
```json
{
    "dependencies":{
        "express": "*"
    },
    "scripts": {
        "start": "node index.js"
    }
}
``` 
- Prepare index.js
```js
const express = require('express');
const app = express();
app.get('/',(req,res) => {
    res.send('hi there');
});
app.listen(8080,()=> {
    console.log('Listening on port 8080');
});
```

42. A few planned errors
- `npm install` will install all dependencies
- `npm start` will start the server
- We assume that npm is installed already
- Flow
    - Specify a base image: FROM alpine
    - Run some commands to install additional programs: RUN npm install
    - Specify a command to run on container startup: CMD["npm","start"]
- Dockerfile
```sh
# base image
FROM alpine
# Install some dependencies
RUN npm Install
# Default command
CMD ["npm","start"]
```
- Command:
```bash
$ sudo docker build .
Sending build context to Docker daemon  4.096kB
Step 1/3 : FROM alpine
 ---> 9c842ac49a39
Step 2/3 : RUN npm Install
 ---> Running in cf5cb5528037
/bin/sh: npm: not found --------------> how to fix?
The command '/bin/sh -c npm Install' returned a non-zero code: 127
```

43. Required Node base image version
- `FROM alpine` -> `FROM 14-alpine`

44. Base image issues
- `/bin/sh: npm: not found` -> because alpine image is very minimal
    - We may select more comprehensive images
    - Or install npm first
- hub.docker.com -> explore -> find base images
- `FROM alpine` -> `FROM node:6.14` : an image with node 6.14
- `FROM alpine` -> `FROM node:alpine` : the most stripped image with node
```bash
$ sudo docker build .
Sending build context to Docker daemon  4.096kB
Step 1/3 : FROM node:alpine
 ---> d1b0127ae8b2
Step 2/3 : RUN npm install
 ---> Running in bf0cf730a622
npm ERR! Tracker "idealTree" already exists

npm ERR! A complete log of this run can be found in:
npm ERR!     /root/.npm/_logs/2022-03-31T23_33_06_189Z-debug-0.log
The command '/bin/sh -c npm install' returned a non-zero code: 1
```
- Now new error message?

45. A few missing files
- How to bring local files to a new container?
    - When npm cannot find package.json

46. Copying build files
- `COPY ./ ./`
    - First argument is the relative folder from the build context
    - Second argument is the absolute folder in the container
```bash
# base image
FROM node:alpine
# Install some dependencies
COPY ./ ./
RUN npm install
# Default command
CMD ["npm","start"]
```
- `docker build -t mydocker/simpleweb .`
- `docker run mydocker/simpleweb`
    - But still 8080 port is not reachable

47. Container port mapping
- Explicit port mapping is necessary for containers
    - Not from Dockerfile but from docker run command
    - `docker run -p A:B image_name`
    - Map the port A of the machine into the port B of the container
- `docker run -p 8080:8080 mydocker/SimpleWeb` then open the localhost:8080 from a web-browser

48. Specifying a working directory
- Ref:https://stackoverflow.com/questions/57534295/npm-err-tracker-idealtree-already-exists-while-creating-the-docker-image-for
```
# base image
FROM node:alpine
WORKDIR /usr/app
# Install some dependencies
COPY ./ ./
RUN npm install
# Default command
CMD ["npm","start"]
```
- Command
```
$ sudo docker build -t mydocker/simpleweb .
Sending build context to Docker daemon  4.096kB
Step 1/5 : FROM node:alpine
 ---> d1b0127ae8b2
Step 2/5 : WORKDIR /usr/app
 ---> Using cache
 ---> 157e5064d949
Step 3/5 : COPY ./ ./
 ---> Using cache
 ---> 9bffbe410060
Step 4/5 : RUN npm install
 ---> Using cache
 ---> 2943ce2dfe79
Step 5/5 : CMD ["npm","start"]
 ---> Using cache
 ---> cc9bddf4f990
Successfully built cc9bddf4f990
Successfully tagged mydocker/simpleweb:latest
hpjeon@hakune:~/hw/class/udemy_docker/simple_web$ sudo docker run -p 8080:8080 mydocker/simpleweb
> start
> node index.js
Listening on port 8080
```
- Open another terminal and run:
```
$ sudo docker ps
[sudo] password for hpjeon: 
CONTAINER ID        IMAGE                COMMAND                  CREATED             STATUS              PORTS                    NAMES
a6333aba6d01        mydocker/simpleweb   "docker-entrypoint.s…"   11 seconds ago      Up 9 seconds        0.0.0.0:8080->8080/tcp   peaceful_matsumoto
```
- We can attach to the existing container
```
$ sudo docker exec -it a6333aba6d01 sh
/usr/app # pwd
/usr/app
```

49. Unnecessary rebuilds
- How to update index.js for the simpleweb container
- Need to rebuild the image
    - But we may skip npm install as it is already done and not affected by index.js

50. Minimizing cache busting and rebuilds
- Update Dockerfile in the following order
- Segment the dependency
```bash
# base image
FROM node:alpine
WORKDIR /usr/app
# Install some dependencies
COPY ./package.json ./
RUN npm install
COPY ./ ./
# Default command
CMD ["npm","start"]
```
- Docker build will check dependency and can use cache, instaed of npm install from scratch
```bash
$ sudo docker build -t mydocker/simpleweb .
Sending build context to Docker daemon  4.096kB
Step 1/6 : FROM node:alpine
 ---> d1b0127ae8b2
Step 2/6 : WORKDIR /usr/app
 ---> Using cache <-------- cached
 ---> 157e5064d949
Step 3/6 : COPY ./package.json ./
 ---> Using cache <-------- cached
 ---> 2808334b7189
Step 4/6 : RUN npm install
 ---> Using cache <-------- cached
 ---> 9afb8e790c63
Step 5/6 : COPY ./ ./
 ---> 523cad0f4a7a <-------- this is from scratch
Step 6/6 : CMD ["npm","start"]
 ---> Running in 7b0616e12fcb
Removing intermediate container 7b0616e12fcb
 ---> ff704b1a096c
Successfully built ff704b1a096c
Successfully tagged mydocker/simpleweb:latest
```

## Coupling multiple containers

51. App overview
- Web sever showing the number of visitors
- Needs 
    - webserver
    - Node app to show interface
    - Redis server recording visitors
- A single container having node app and Redis
    - For multiple web servers, we better split node app and Redis server
- So many node app containers with one Redis server
- We practice one container of node and another container of Redis

52. App server stater code
- package.json
```json
{
    "dependencies": {
        "express": "*",
        "redis":"2.8.0"
    },
    "scripts": {
        "start": "node index.js"
    }
}
```
- index.js
```js
const express = require('express');
const redis = require('redis');
const app = express();
const client = redis.createClient();
client.set('visits',0)
app.get('/', (req,res) => {
    client.get('visits', (err,visits)=> {
        res.send('Number of visits is ' + visits);
        client.set('visits', parseInt(visits)+1);
    });
});
app.listen(8081, ()=> {
    console.log('Listening on port 8081');
})
```

53. Assembling a Dockerfile
```bash
FROM node:alpine
WORKDIR '/app'
COPY package.json .
RUN npm install
COPY . .
CMD ["npm","start"]
```
- Command:
```bash
$ sudo docker build -t mydocker/visits:latest .
Sending build context to Docker daemon  4.096kB
Step 1/6 : FROM node:alpine
 ---> d1b0127ae8b2
Step 2/6 : WORKDIR '/app'
 ---> Using cache
 ---> d6f966897ab6
Step 3/6 : COPY package.json .
 ---> Using cache
 ---> 6e904874b0c8
Step 4/6 : RUN npm install
 ---> Using cache
 ---> f4ed38094d86
Step 5/6 : COPY . .
 ---> Using cache
 ---> d83f8c473b45
Step 6/6 : CMD ["npm","start"]
 ---> Using cache
 ---> 86852eeb2d11
Successfully built 86852eeb2d11
Successfully tagged mydocker/visits:latest
```

54. Introducing Docker Compose
- Trial command
```bash
$ sudo docker run mydocker/visits
> start
> node index.js
Listening on port 8081
node:events:505
      throw er; // Unhandled 'error' event
      ^
Error: connect ECONNREFUSED 127.0.0.1:6379
    at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1187:16)
```
- Error as there is no redis
- Let's run redis image
```bash
$ sudo docker run redis
Unable to find image 'redis:latest' locally
latest: Pulling from library/redis
```
- Open another terminal and run mydocker/visits
- Still fails. Why?
- There is no automatic communication b/w different images
    - How to couple them?
    - Use Docker CLI: not recommended though due to complexity
    - Use Docker Compose
- Docker compose
    - Separate CLI from Docker
    - Can start up multiple docker containers
    - Automates complex commands

55. Docker compose files
- Prepare docker-compose.yml
```yml
version: '3'
services: 
  redis-server:  # this name will be used in index.js
    image: 'redis' # we use an image from docker hub
  node-app:
    build: .  # we build new image from the current folder
    ports:
      - "8081:8081"
```

56. Networking with Docker compose
- We don't need to update port b/w containers
    - Configure redis-node connection in the index.js
```js
const express = require('express');
const redis = require('redis');
const app = express();
const client = redis.createClient({
    host: 'redis-server',  //--> this is the name from docker-compose.yml
    port: 6379
});
client.set('visits',0)
app.get('/', (req,res) => {
    client.get('visits', (err,visits)=> {
        res.send('Number of visits is ' + visits);
        client.set('visits', parseInt(visits)+1);
    });
});
app.listen(8081, ()=> {
    console.log('Listening on port 8081');
})
```    

57. Docker compose commands
- `docker run image` -> `docker-compose up`
- `docker build. ; docker run image` -> `docker-compose up --build`
- Command
```bash
$ sudo docker-compose up
Creating network "visits_default" with the default driver
Building node-app
Step 1/6 : FROM node:alpine
 ---> d1b0127ae8b2
Step 2/6 : WORKDIR '/app'
 ---> Using cache
 ---> d6f966897ab6
Step 3/6 : COPY package.json .
 ---> 46647ab0a49c
Step 4/6 : RUN npm install
 ---> Running in 242fe31ccb9e
...
redis-server_1  | 1:M 01 Apr 2022 17:47:26.236 * Ready to accept connections
visits_node-app_1 exited with code 1
```
- Open localhost:8081 from a web-browser
- ~~When old index.js keeps appearing, run `sudo docker system prune -a`~~
- To update index.js, run `sudo docker-compose up --build`

58. Stopping Docker compose containers
- Running docker compose
    - `sudo docker-compose up`
    - `sudo docker-compose up -d` running as background
- Stopping docker compose
    - `sudo docker-compose down`

59. Container maintenance with compose
- Make a container crash
```js
const express = require('express');
const redis = require('redis');
const app = express();
const process = require('process');
const client = redis.createClient({
    host: 'redis-server',
    port: 6379
});
client.set('visits',0)
app.get('/', (req,res) => {
    process.exit(0); <---- make the server crash
    client.get('visits', (err,visits)=> {
        res.send('Number of visits is ' + visits);
        client.set('visits', parseInt(visits)+1);
    });
});
app.listen(8081, ()=> {
    console.log('Listening on port 8081');
})
```
- Command:
```bash
$ sudo docker-compose up --build
Creating network "visits_default" with the default driver
Building node-app
Step 1/6 : FROM node:alpine
 ---> d1b0127ae8b2
...
redis-server_1  | 1:M 01 Apr 2022 18:20:46.024 * Ready to accept connections
visits_node-app_1 exited with code 0 <---- when a browser calls the localhost, it crashes
```
- But docker is still running
```bash
$ sudo docker ps
CONTAINER ID   IMAGE     COMMAND                  CREATED              STATUS              PORTS      NAMES
32e5fdce9245   redis     "docker-entrypoint.s…"   About a minute ago   Up About a minute   6379/tcp   visits_redis-server_1
```
- How to fix?

60. Automatic container restarts
- Restart policies
    - "no": Default. Double-quotes are mandatory!!!
        - yaml will understand `no` as `false`
    - always:
    - on-failure: will ignore return 0
    - unless-stopped
- In the docker-compose.yml
```yml
version: '3'
services: 
  redis-server: 
    image: 'redis'
  node-app:
    restart: always <------------ Added
    build: .
    ports:
      - "8081:8081"
```
- Test again
```bash
node-app_1      | Listening on port 8081 
visits_node-app_1 exited with code 0  <---------- crash
node-app_1      | 
node-app_1      | > start  <--------- start again
node-app_1      | > node index.js
node-app_1      | 
node-app_1      | Listening on port 8081
```
- When on-failure and process.exit(0) in the index.js, crash will not restart the container

61. Container status with Docker compose
```bash
$ sudo docker-compose ps
        Name                    Command           State           Ports         
--------------------------------------------------------------------------------
visits_node-app_1       docker-entrypoint.sh      Up      0.0.0.0:8081->8081/tcp
                        npm start                         ,:::8081->8081/tcp    
visits_redis-server_1   docker-entrypoint.sh      Up      6379/tcp              
                        redis ...                             
```

62. Development workflow
- Develpment -> Testing -> Deployment
    - Repeats

63. Flow specifics
- Push to github -> Pull request to Master branch -> Travis CI for testing -> Deployment to AWS

64. Docker's purpose
- No docker in workflow

65. Project generation
- React app project
- Needs node.js

66. Create React App generation
- Upgrade of node.js : https://askubuntu.com/questions/426750/how-can-i-update-my-nodejs-to-the-latest-version
```
sudo npm cache clean -f
sudo npm install -g n
sudo n stable
```

67. More on project generation
- npx create-react-app frontend
    - Will take time to download necessary components

68. Necessary commands
- cd frontend
- npm run test
    - tests if the app works
- npm run build
    - Will produce build folder
- npm run start
    - A react server and a web browser will open
    - Find the service at localhost:3000

69. Creating the Dev Docker file
- Dockerfile.dev: for the development
    - A separate Dockerfile will be made for build 
``` bash
FROM node:16-alpine
WORKDIR '/app'
COPY package.json .
RUN npm install 
COPY . .
CMD ["npm", "run", "start"]
```
- Command: docker build -f Dockerfile.dev .

70. Duplicated dependencies
- The above process installs all the dependencies shown in frontend/node_modules folder
    - Delete node_modules folder

71. Starting the container.
- 
