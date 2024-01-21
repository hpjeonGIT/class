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

16. Handouts

## Section 3: Ansible Automation with Simple Playbooks

17. YAML File syntax
- Sequential order
- Each task is processed one at a time
- Indentation is important. No tab. Only spaces

18. YAML File Syntax Example

19. Creating First Playbook

20. Output Playbook

21. Multiple Tasks Playbook

22. Playbook for installing and starting a service

23. Handouts

## Section 4: Ansible Automation for Remote clients

## Section 5: Ansible Automation with ad-hoc tasks

## Section 6: Advance Ansible Automation Features

## Section 7: Additional Features in Ansible

## Section 8: Securing Ansible

## Section 9: Ansible Management Toolks

## Section 10: Ansible Resources
