## JENKINS Beginner Tutorial - Step by Step
- Instructor:  Raghav Pal

1. Introduction and Getting Started
- https://www.jenkins.io/doc/book/installing/linux/#debianubuntu
```
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo tee \
  /usr/share/keyrings/jenkins-keyring.asc > /dev/null
echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null
sudo apt-get update
sudo apt-get install jenkins
```
- Or locally download LTS versioon of jenkins.war and run command 
  - `java -jar jenkins.war`
  - Will generate passwd
  - Or find the passwd at .jenkins/secrets/initialAdminPassword
    - username: `admin`
  - Goto a browser: http://localhost:8080
  - Select and click suggested plugins
  - Continue as an admin
- Configuration of java
  - javac was running from 11 while java is from jre of java8
  - update-alternatives --config java
  - update-alternatives --config javac
  - Sync using `sudo update-alternatives --config java`

2. How to change Home directory of Jenkins
- `~/.jenkins`
- Manage Jenkins -> Configure System -> Home directory

3. Using CLI
- When to change port: `java -jar jenkins.war --httpPort 9090`
- Goto `http://localhost:8080/cli/` and download `jenkins-cli.jar`
- `java -jar jenkins-cli.jar -s http://localhost:8080 -webSocket -auth admin:XXXXX help`


4. How to create Users + Manage + Assign Roles
- Create User from GUI menu 
  - user1:Abc123
  - user2 for tester
- admin-> manage Jenkins -> Plugin Manager -> Install Role-based authorization strategy 
  - Install and restart
  - Now Dashboard -> Configure Global Security -> Authorizatoin shows Role-based strategy. Click and Apply
  - Now user1 cannot login
  - As admin, Dashboard -> Manage and Assign Roles -> Manage Roles
    - Add `employee` role and apply as necessary on Global 
    - Add `developer:Dev.*` in Item roles
    - Add `tester:Test.*` in Item roles
    - Apply and Save
  - Assign Roles
- Dashboard -> New item : this is new project  

5. Basic Configurations
- Manage Jenkins -> Configure system
- Home directory: where jenkins configuration is stored
- Root directory: the root folder of each item (or project)
- Labels -> can couple with each item (General-> restrict where the project can be run)

6. Getting started with JOBs
- Build periodically
  - similar to cron tab
  - `*****`: every minute
- Build:
  - bash default (?)
- Post-build Actions
  - can chain another build  
- Trigger the build remotely : http token

7. Jenkins integration with GIT (SCM)
  - Create a java program
```java
public class Hello {
  public static void main(String[] args) {
    for(int i=1;i<=10;i++) {
      System.out.println("Hello world ... "+i);
    }
  }
}
```
  - javac Hello.java 
  - java Hello
- New item as HelloWorld
  - Build as Execute shell
```bash
cd /home/hpjeon/hw/class/udemy_jenkins
javac Hello.java
java Hello
```
- Coupling with git
  - Create a project HelloWorld in github
  - Jenkins -> Dashboard -> Manage Plugins -> find Git plugin
    - Already installed when suggested plugin is installed
  - Item -> HelloWorld -> Configure -> General -> Souirce Code management -> Git -> Enter github page
    - Build Triggers -> Poll SCM
- Q:
  - The instructor coupled the source files at the local disk with git repo -> any change by committing from local drive will trigger SCM, running code locally. But how to download from github and run, without local repo?

8. What is automated deployment
- Build -> Deploy -> Test -> Release

9. How to do Automated Deployment
- Deploy plugin 
- Coupling with Tomcat

10. Notifications - How to send Email from Jenkins
- For smtp server, smtp.gmail.com:465

11. What is pipeline in Jenkins (DevOps)
- A workflow with group of events or jobs that are chained and integrated with each other in sequence

12. How to setup DELIVERY PIPELINE in Jenkins (Step by Step)

13. How to setup BUILD PIPELINE in Jenkins (Step by Step)

14. How to Parameterize job in jenkins

15. How to run job from command line

16. How to run parameterized job from comman dline

17. How to create parameters - Check box, drop down, radio button

18. How to Pass Parameters to downstream job

19. How to create Build monitor view in jenkins

20. How to setup Jenkins on Tomcat


## Practice on Jenkins
- Using https://github.com/hpjeonGIT/PracticeJenkins
  - No credential. Open source
- Branch specifier: */main
- Build Triggers
  - Poll SCM with schedule of `* * * * *`
- Build 
  - Execute shell
```bash
cd java
javac HelloWorld.java
java HelloWorld
```
- Manual modification from a browser triggers SCM and Jenkins clones successfully
```
Started by user admin
Running as SYSTEM
Building in workspace /home/hpjeon/.jenkins/workspace/HelloWorld
The recommended git tool is: NONE
No credentials specified
 > git rev-parse --resolve-git-dir /home/hpjeon/.jenkins/workspace/HelloWorld/.git # timeout=10
Fetching changes from the remote Git repository
 > git config remote.origin.url https://github.com/hpjeonGIT/PracticeJenkins # timeout=10
Fetching upstream changes from https://github.com/hpjeonGIT/PracticeJenkins
 > git --version # timeout=10
 > git --version # 'git version 2.17.1'
 > git fetch --tags --progress -- https://github.com/hpjeonGIT/PracticeJenkins +refs/heads/*:refs/remotes/origin/* # timeout=10
 > git rev-parse refs/remotes/origin/main^{commit} # timeout=10
Checking out Revision 3441151887a5d1c160a91420f1ab0a6979c8f279 (refs/remotes/origin/main)
 > git config core.sparsecheckout # timeout=10
 > git checkout -f 3441151887a5d1c160a91420f1ab0a6979c8f279 # timeout=10
Commit message: "3rd"
 > git rev-list --no-walk 0fd5a17d3b8b12b7aca9905d584a8fa21326e1ab # timeout=10
[HelloWorld] $ /bin/sh -xe /tmp/jenkins9891791197961895954.sh
+ cd java
+ javac HelloWorld.java
+ java HelloWorld
Hello world 3rd edit... 1
Hello world 3rd edit... 2
Hello world 3rd edit... 3
Hello world 3rd edit... 4
Hello world 3rd edit... 5
Hello world 3rd edit... 6
Hello world 3rd edit... 7
Hello world 3rd edit... 8
Hello world 3rd edit... 9
Hello world 3rd edit... 10
Finished: SUCCESS
```
