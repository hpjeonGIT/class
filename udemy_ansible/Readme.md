## Title: Complete DevOps Ansible Automation Training
- Instructor: Imran Afzal

## Section 1: Introduction

1. Introduction

2. Course Overview

3. What is Ansible?
- Provisioning
  - Baremetal
  - VM
  - Network devices
  - Storage
  - Cloud Platform
- Configuration
   - Update/upgrade
   - Package installation
   - Service configuration - stop/start/restart
   - User/groups
   - Permission
  - Backup
  - Application deployment
  - Weekly/monthly reboots
  - Orchestration

4. Brief history of Ansible

5. Benefits of Ansible
- Agentless
- Open source
- Avoid human errors
- Automation
- Higher productivity
- Easy to use

6. Terminologies in Ansible
- Control node/Ansible server: server runs Ansible
- Modules: commands to be executed on client side
- Task: collection of modules
- Playbook: Automation file with step-by-step execution of multiple tasks
  - Written in YAML
- Inventory: File that has information of remote clients
- Tag: a reference or alias to a specific task  
- Role: Splitting of playbook into smaller groups

7. How Ansible works?
- Each specific task in Ansible is written through a Module or modules
- Multiple modules are written in sequential order
- Multiple modules for related Tasks is called a Play
- All Plays together makes a Playbook
- Playbook is written with YAML
- Commands example
  - Run modules through yaml: `ansible-playbook example.yml`
  - Run module independently: `ansible myservers -m ping`
- Ansible Configuration files
  - /etc/ansible/ansible.cfg
  - /etc/ansible/hosts
  - /etc/ansible/roles

8. Other Automation Tools
- Puppet and Chef

9. Free Source Ansible and Red Hat Ansible

10. Handouts

## Section 2: Lab Design and Setup

11. Lab design

12. Installing Virtualization SW
- We use VirtualBox

13. Creating a VM and Installing Linux

14. Creating Ansible Clients
- Download VB image from https://www.linuxvmimages.com/

15. Installing Ansible
- Take VM snapshot
  - Virtual Box -> Take -> snapshot
- https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html
- yum install epel-release
- yum install ansible # + ansible-doc for RHEL8
- ansible --version
- ansible localhost -m ping # testing for the local machine
- Ansible config files
```bash
/etc/ansible # Default directory
/etc/ansible/ansible.cfg
/etc/ansible/hosts
/etc/ansible/roles
```
- On Ubuntu, sudo apt install ansible

16. Handouts

## Section 3: Ansible Automation with Simple Playbooks

17. YAML File syntax
- Sequential order
- Each task is processed one at a time
- Indentation is important. No tab. Only spaces

18. YAML File Syntax Example

19. Creating First Playbook
- su - root
- mkdir /etc/ansible/playbooks
- cd /etc/ansible/playbooks
- vi first.yml
```yaml
---
- name: "My first playbook"
  hosts: localhost

  tasks:
  - name: "test connectivity"
    ping:
```
- Check the syntax
  - ansible-playbook --syntax-check first.yml
- Or dry run
  - ansible-playbook --check first.yml
- Run the playbook
  - ansible-playbook /root/ansible/first.yml
- Demo:
```bash
$ ansible-playbook first.yml
/usr/lib/python3/dist-packages/paramiko/transport.py:219: CryptographyDeprecationWarning: Blowfish has been deprecated
  "class": algorithms.Blowfish,
[WARNING]: provided hosts list is empty, only localhost is available. Note that
the implicit localhost does not match 'all'
...

PLAY [My first playbook] *******************************************************

TASK [Gathering Facts] *********************************************************
ok: [localhost]

TASK [test connectivity] *******************************************************
ok: [localhost]

PLAY RECAP *********************************************************************
localhost                  : ok=2    changed=0    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
```

20. Output Playbook
- cd /etc/ansible/playbooks
- vi hello.yml
```yaml
---
- name: "My second playbook"
  hosts: localhost

  tasks:
  - name: Print Hello World
    debug: msg="Hello World"
```
- Demo:
```bash
$ ansible-playbook helloworld.yml 
/usr/lib/python3/dist-packages/paramiko/transport.py:219: CryptographyDeprecationWarning: Blowfish has been deprecated
  "class": algorithms.Blowfish,

PLAY [My second playbook] ******************************************************

TASK [Gathering Facts] *********************************************************
ok: [localhost]

TASK [Print Hello World] *******************************************************
ok: [localhost] => {
    "msg": "Hello World"
}

PLAY RECAP *********************************************************************
localhost                  : ok=2    changed=0    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   

```

21. Multiple Tasks Playbook
```yml
- name: 2 tasks
  hosts: localhost

  tasks:
  - name: Test connectivity
    ping:

  - name: Print Hello World
    debug: msg="Hello World"
```
- Demo:
```bash
$ ansible-playbook mtask.yml 

PLAY [2 tasks] *****************************************************************

TASK [Gathering Facts] *********************************************************
ok: [localhost]

TASK [Test connectivity] *******************************************************
ok: [localhost]

TASK [Print Hello World] *******************************************************
ok: [localhost] => {
    "msg": "Hello World"
}

PLAY RECAP *********************************************************************
localhost                  : ok=3    changed=0    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
```

22. Playbook for installing and starting a service
```yml
---
- name: Installing and Running apache
  hosts: localhost
  tasks:
  - name: Install apache
    yum:
     name: httpd
     state: present
  - name: start httpd
    service: 
     name: httpd
     state: started
```
- At ubuntu, error message like `An exception occurred during task execution. To see the full traceback, use -vvv. The error was: AttributeError: module 'lib' has no attribute 'X509_V_FLAG_CB_ISSUER_CHECK'`
  - https://stackoverflow.com/questions/73830524/attributeerror-module-lib-has-no-attribute-x509-v-flag-cb-issuer-check
  - sudo python3 -m pip install pip --upgrade
  - sudo pip install pyopenssl --upgrade
  - ansible-playbook --check -v  ./apache.yml 

23. Handouts

## Section 4: Ansible Automation for Remote clients

24. Welcome to Ansible Automation for Remote Clients

25. Remote Clients hosts File Syntax
- /etc/ansible/hosts
   -Using `-i`, you can specify the locatin of hostfile, if it is located differently
- Default is all of hosts (group name is all)
- Can use [:] for the range of numbers
```bash
[appserver]
app1.ex.com
app2.ex.com

[webserver]
web1.ex.com
web2.ex.com
10.1.2.[3:15]
```
- `ansible-inventory --list`: shows the listed host file

26. Establish Connection to Remote Clients
- Add ip address of clients into /etc/ansible/hosts and check `ansible-inventory --list`
- Build ssh connection
  - ssh-keygen
  - ssh-copy-id <client_ip>
- ansible all -m ping
- Seems failing. Why?

27. Playbook - Check Remote Clients Connectivity

28. Playbook - Copy Files to Remote Clients
- Using copy module
- copy.yml
```yml
---
- name: Copy file from local to remote clients
  hosts: all
  tasks:
  - name: Copying file
    become: true
    copy:
     src: /some_folder/some.cfg
     dest: /tmp
     owner: hpjeon
     group: localuser
     mode: 0644
```
- ansible-playbook copy.yml

29. Playbook - Change File Permission
- Changing file mode of existing files in the remote clients, using file module
- filepermission.yml
```yml
---
- name: Change file permissions
  hosts: all
  tasks:
  - name: File Permissions
    file:
     path: /some_folder_in_clients
     mode: a+w # or 0644
```

30. Playbook - Setup Apache and Open Firewall Port
- Demo:
    - Install httpd package
    - Start httpd service
    - Open http service port in firewall
    - Restart firewalld service
- Requires additional ansible collection for firewalld
    - ansible-galaxy collection install ansible.posix
- demo30.yml
```yml
---
- name: Setup httpd and open firewall pott
  hosts: all
  tasks:
  - name: Install apache package
    yum:
     name: httpd
     state: present  # absent to remove, latest to upgrade
- name: start httpd
   service:
    name: httpd
    state: started  #started| stoppped| reloaded| restarted
- name: Open port 80 for http access
   firewalld:
    service: http
    permanent: true
    state: enabled
- name: Restart firewalld service to load firewall changes
   service:
    name: firewalld
    state: reloaded
```

31. Playbook - Run Shell Scripts on Remote Clients
- Running existing shell scripts on the remote clients
```yml
---
- name: Playbook for shell script
  hosts: all
  tasks:
  - name: Run shell script
    shell: "/some_folder/somescript.sh"
```
- The script will be executed from home folder of root (or ssh account). May use absolute path for intermediate/results output

32. Playbook - Schedule a job (crontab)
- Setting a cronjob
```yml
---
- name: Create a cron job
  hosts: all
  tasks:
  - name: Schedule cron
    cron:
     name: This is scheduled by Ansible
     minute: "0"
     hour: "10"
     day: "*"
     month: "*"
     weekday: "4"
     user: root
     job: "/somefolder/somescript.sh"
```

33. Playbook - User Account Management
- Create a user george and configure home folder/bash
```yml
---
- name: Playbook for creating users
  hosts: all
  tasks:
  - name: Create users
    user:
     name: george
     home: /home/george
     shell: /bin/bash
```    

34. Playbook - Add or Update User Password
- Ansible does not allow us to pass a text password through module
    - Hash using sha512 and sent from command line
```yml
---
- name: Add or update user password
  hosts: all
  tasks:
  - name: Change "george" password
    user:
     name: george
     update_password: always
     password: "{{ newpassword | password_hash{'sha512') }}" # newpassword is fed from a line comand
```
- ansible-playbook changepasswd.yml --extra-vars newpassword=XXXX

35. Playbook - Download Package from a URL
- Download tomcat from non-package repo. Localhost only
```yml
---
- name: Download Tomcat from tomcat.apache.org
  hosts: localhost
  tasks:
  - name: Create a directory /opt/tomcat
    file:
     path: /opt/tomcat
     state: directory
     mode: 0755
     owner: root
     group: root
  - name: Download Tomcat using get_url
    geturl:
     url: https://dlcdn.apache.org/tomcat/tomcat-8/v8.5.78/bin/apache-tomcat-8.5.78.tar.gz
     dest: /opt/tomcat
     mode: 0755
     group: hpjeon
     owner: hpjeon
```

36. Playbook - Kill a Running Process
- Find a running process by name
- Ignore any errors
```yml
---
- name: Find a process and kill it
  hosts: 10.x.x.1
  tasks:
  - name: Get running processes from remote hosts
    ignore_errors: yes
    shell: "ps -few |grep top| awk '{print $2}'" # we kill top process in the remote client
    register: running_process
  - name: Kill running processes
    ignore_errors: yes
    shell: "kill {{ item }}"
    with_items: "{{ running_process.stdout_lines }}"
```   

37. Pick and Choose Steps
- ansible-playbook http.yml --start-at-task 'Install XXX'

38. Create and Mount New Storage
- Modules of parted and mount for storage creation/mount
    - ansible-galaxy collection install community.general
    - ansible-galaxy collection install ansible.posix
```yml
---
- name: Create and mount new storage
  hosts: all
  tasks:
  - name: create new partition
    parted:
     name: files
     label: gpt
     device: /dev/sdb
  - name: Create xfs filesystem
    filesystem:
     dev: /dev/sdb1
     fstype: xfs
  - name: Create mount directory
    file:
     path: /data
     state: directory
  - name: mount the filesystem
    mount:
     src: /dev/sdb1
     fstype: xfs
     state: mounted
     path: /data
```    

39. Handouts

## Section 5: Ansible Automation with ad-hoc tasks

40. Welcome ansible Automation with Ad-hoc Tasks

41. Ansible Ad-Hoc Commands (part 1)
- Ad-hoc commands run on as needed and usually those tasks don't repeat
- ansible [target] -m [module] -a "[module options]"
    - ansible localhost -m ping
    - ansible all -m file -a "path=/somefolder/somefile  state=touch mode=700" # create a file on all remote clients
    - ansible all -m file -a "path=/somefolder/somefile state=absent" # delete a file on all remote clients
    - ansible all -m copoy -a "src=/somefolder/src1 dest=/somefolder/target" # copy a file from ansible control node to remote clietns

42. Ansible Ad-Hoc Commands (part 2)
- ansible all -m yum -a "name=telnet state=present"
- ansible all -m yum -a "name=httpd-manual state=present"
- ansible all -m service -a "name=httpd state=started"
- ansible all -m service -a "name=httpd state=started enabled=yes" # start httpd everytime when reboot
- ansible all -m shell -a "systemctl status httpd" # run command on the remote clients
- ansible all -m yum -a "name=httpd state=absent" # remove httpd package
- ansible all -m shell -a "yum remove httpd"

43. Ansible Ad-Hoc Commands (part 3)
- ansible all -m user -a "name=jsmith home=/home/jsmith shell=/bin/bash state=present"
- ansible all -m user -a "name=jsmith group=vgluser" # adding a group
- ansible all -m user -a "name=jsmith home=/home/jsmith shell=/bin/bash state=absent"
- ansible all -m shell -a "userdel jsmith"
- ansible all -m setup # Info of clients
- ansible client1 -a "/sbin/reboot" # reboot client1 node
44. Handhouts

## Section 6: Advance Ansible Automation Features

45. Roles
- Roles simlifies long playbooks by grouping tasks into smaller playbooks
- The roles are the way of breaking a playbook into multiple playbook files. This simplifies writing complesx playbooks, and it makes them easier to reuse
- Writing ansible code to manaage the same service for multiple environments creates more complexity and it becomes difficult to manage everything in one ansible playboook. Also sharing code among other teams becomes difficutl.
- Roles are like templates that are most of the time static and can be called by the playbooks
- Roles allow the entire configuration to be grouped in:
    - Tasks
    - Modules
    - Variables
    - Handlers
- Scenario:
    - east-webservers need 1) intall http 2) start httpd 3) open port on firewall 4) restart firewall
    - west-webservers need 1) intall http 2) start httpd
    - Defining tasks per hostname will be lengthy
```yml
- name: Full install
  hosts: east-webservers
  roles:
  - fullinstall
- name: Basic install
  hosts: west-webservers
  roles:
  - basicinstall
```   
- To create roles
    - /etc/ansible/roles of control node
    - Make directory for each role
    - mkdir /etc/ansible/roles/basicinstall; mkdir /etc/ansible/roles/basicinstall/tasks; touch /etc/ansible/roles/basicinstall/main.yml
    - mkdir /etc/ansible/roles/fullinstall;  mkdir /etc/ansible/roles/fullinstall/tasks;  touch /etc/ansible/roles/fullinstall/main.yml
    - fullinstall/tasks/main.yml
```yml
---
- name: Install apache package
  yum:
   name: httpd
   state: present  # absent to remove, latest to upgrade
- name: start httpd
  service:
   name: httpd
   state: started
- name: Open port 80 for http access
  firewalld:
   service: http
   permanent: true
   state: enabled
- name: Restart firewalld service to load firewall changes
  service:
   name: firewalld
   state: reloaded
```
    - basicinstall/tasks/main.yml
```yml
---
- name: Install apache package
  yum:
   name: httpd
   state: present  # absent to remove, latest to upgrade
- name: start httpd
  service:
   name: httpd
   state: started
```

46. Roles by Application
```yml
---
- name: Install packages
  hosts: all
  tasks:
  - name: Install Apache package
    yum:
     name: httpd
     state: present
  - name: Install Time package
    yum
     name: ntpd or chrony
     state: present
  - name: Install DNS package
    yum
     name: named
     state: present
```
- Update the above package installations as roles
    - /etc/ansible/roles/apache/tasks/main.yml
```yml
- name: Install Apache package
  yum:
   name: httpd
   state: present
```
    - /etc/ansible/roles/ntpd/tasks/main.yml
```yml
- name: Install Time package
  yum
   name: ntpd or chrony
   state: present
```
    - /etc/ansible/roles/named/tasks/main.yml
```yml
- name: Install DNS package
  yum
   name: named
   state: present
```
- Playbook yaml
```yml
---
- name: Install packages
  hosts: all
  roles:
  - apache
  - ntpd
  - named
```

47. Roles on Ansible Galaxy
- www.galaxy.ansible.com

48. Tags
- Instead of running an entire playbook, we may use tags to run a specific task using a tag
```yml
---
- name: Setup httpd and open firewall pott
  hosts: all
  tasks:
  - name: Install apache package
    yum:
     name: httpd
     state: present  # absent to remove, latest to upgrade
    tags: i-httpd
- name: start httpd
   service:
    name: httpd
    state: started  #started| stoppped| reloaded| restarted
   tags: s-httpd
```
- ansible-playbook myplay.yml -t i-httpd
- ansible-playbook myplay.yml -t s-httpd
- ansible-playbook myplay.yml --list-tags # show the list of tags in the yaml

49. Variables
- Variables are like containers that hold the defined value which can be used repetitively
- Name can include letters, numbers and underscore
- Starting with a letter
- N space, dots, hypen
```yml
---
- name: Install somepackge
  hosts: all
  vars:
   newpackage: httpd
  tasks:
  - name: Install package
    yum:
     name: "{{ newpackage}}"
     state: present
- name: start package
   service:
    name: "{{ newpackage}}"
    state: started
```

50. Variables in Inventory File
- Filest at the control node
    - hosts file

51. Handouts

## Section 7: Additional Features in Ansible

52. Handlers
- Handlers are executed at the end of the play once all tasks are finished
- Basically handlers are tasks that onlyrun when notified
- Each handler must have globally unique name   
```yml
---
- name: verify apache installation
  hosts: localhosts
  tasks:
  - name: Ensure apache is the latest or not
    yum:
     name: httpd
     state: latest
- name: Copy updated config file
   copy:
    src: /tmp/httpd.conf
    dest: /etc/httpd.conf
   notify:
   - Restart apache # See below handlers-> name. It doesn't mean it will jump to handlers section now
- name: Ensure apache is running
   services:
    name: httpd
    state: started # now all tasks are completed. Then Handlers are executed
handlers:
- name: Restart apach
   service:
    name: httpd
    state: restarted
```

53. Conditions
- Condition execution allows ansible to take actions on its own conditions
```yml
---
- name: Install Apache webserver
  hosts: localhost
  tasks:
  - name: Install Apache on Ubuntu
    apt-get:
     name: apache2
     state: present
    when: ansible_os_family == "Ubuntu"
  - name: Install Apache on CENTOS
    yum:
     name: httpd
     state: present
    when: ansible_os_family == "RedHat"
```
    - `ansible_os_family` is an ansible built-in variable
    - `ansible localhost -m setup` shows all built-in variables

54. Loops 
- Changing permission on many files
- Creating many users
- Installing many packages
- Loop runs until a condition is met
- `loop` and `with_*` keywords
- Generic script for many user creation:
```yml
---
- name: Create users
  hosts: localhost
  tasks:
  - name: Create jerry
    user:
     name : jerry
  - name: Create kramer
    user:
     name : kramer
  - name: Create eliane
    user:
     name : eliane
```
- Using loop:
```yml
---
- name: Create users through loop
  hosts: localhost
  tasks:
  - name: Create jerry
    user:
     name: "{{ item }}"
    loop:
    - jerry
    - kramer
    - eliane
```
- Using with_items and variables:
```yml
---
- name: Create users through loop and var
  hosts: localhost
  vars:
   users: [jerry,kramer,eliane]
  tasks:
  - name: Create users
    user:
     name: '{{ item }}'
    with_items: '{{ users }}'
```
- Demo for multiple packages install
```yml
---
- name: Install packages through var and with_items
  hosts: localhost
  vars:
      packages: [ftp,telnet,htop]
  tasks:
  - name: Install package
    yum:
        name: '{{ items }}'
        state: present
    with_items: '{{ packages }} '
```   
- Demo2 for multiple packages install
```yml
---
- name: Install packages through var and with_items
  hosts: localhost
  vars:
      packages: [ftp,telnet,htop]
  tasks:
  - name: Install package
    yum:
        name: '{{ packages }}'
        state: present
```   

55. Handouts

## Section 8: Securing Ansible

56. Ansible Vault
- Use ansible vault for password protect
- yaml file with `ansible-vault` command
- answible-vault create some.yaml
    - Will ask new passwrord then enter
    - vi editor opens (not vim)
    - Enter contents and save/exit
    - File will be encrypted and cannot see ASCII contents
- ansible-playbook some.yaml --ask-vault-pass
    - Enter passwords then executes
- To read encrypted yaml, ansible-vault view some.yaml
- To edit encrypted yaml, ansible-vault edit some.yaml
- To encrypt regular yaml, ansible-vault encrypt regular.yaml

57. Encrypt Strings within a Playbook
- Only specific string/words can be encrypted
- ansible-vault encrypt_string httpd
- ansible-vault create/encrypt some.yml

58. Handouts

## Section 9: Ansible Management Toolks

59. Welcome to Ansible Management Tools
- Ansible AWX
- Ansible Tower

60. Ansible AWX
- Web application with user interface, REST API, and task engine
- Open source
- Pros
    - Full enterprise features and functionality of Tower
    - Free download
    - No limits of node number
- Cons
    - No technical support from RHEL
    - Not recommended by RHEL
- Deployed as a docker container

61. Ansible Tower
- Commercial version of AWX
- Standard/premium

62. Handouts

## Section 10: Ansible Resources

63. Ansible Additional Commands
- ansible: for ad-hoc command
- ansible-playbook: to run yaml
- ansible-vault: for encryption
- ansible-config: shows/modifies configuration
- ansible-connection: connection command for the remote clients
- ansible-console: Can modify configuration at run time.
- ansible-doc:
    - ansible-doc -l # lists all modules
- ansible-galaxy:
- ansible-inventory: details of host inventory files
    - ansible-inventory -i hosts --graph
    - ansible-inventory --list

64. Ansible Documentation
- https://docs.ansible.com/ansible/latest

65. Community Help

66. Handouts

67. Congratulations

68. Bonus lecture
