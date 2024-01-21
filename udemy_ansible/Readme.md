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

29. Playbook - Change File Permission

30. Playbook - Setup Apache and Open Firewall Port

31. Playbook - Run Shell Scripts on Remote Clients

## Section 5: Ansible Automation with ad-hoc tasks

## Section 6: Advance Ansible Automation Features

## Section 7: Additional Features in Ansible

## Section 8: Securing Ansible

## Section 9: Ansible Management Toolks

## Section 10: Ansible Resources
