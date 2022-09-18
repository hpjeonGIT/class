## Master Puppet for DevOps Success
- Instructor: Craig Dunn

## Section 1: Introduction

1. Introduction

## Section 2: Module 1: Getting Started with Puppet

2. Why Puppet
  - Automation general
    - Cobbler/Kicstart
    - Golden Images
    - Bash scripts
    - Anything repeatable
  - Configuration management
    - Define a desired state
    - For old/new systems
    - Configuration drift
  - Traditional configuration management process
    - Provision -> Configure -> Desired state -> Drift -> Enforce -> Configure again
  - Introducing Puppet
    - Industry standard for config management
    - Define a desired state as code
    - Analyse and compare existing state
    - Enforce compliance
    - Report

3. Installing Puppet
  - 2 VMs
    - puppet.localdomain
    - agent.localdomain
  - Ensure they are netowkred with resolvable hostnames
  - Use httP://vagrantup.com
  - Vagrantfile:
```
$ more Vagrantfile 
# -*- mode: ruby -*-
# vi: set ft=ruby :
#
VAGRANTFILE_API_VERSION = "2"
Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  config.vm.box = "centos/7"
  config.vm.define 'puppet' do |puppet|
    puppet.vm.hostname = 'puppet.localdomain'
    puppet.vm.network "private_network", type: "dhcp"
    puppet.vm.provider :virtualbox do |v|
      v.memory = 2048 #4096
    end
    puppet.vm.provision "shell", inline: %Q{
      cp -r /vagrant/.vim* /root
      rpm -Uvh https://yum.puppetlabs.com/puppetlabs-release-pc1-el-7.noarch.rpm
      yum -y install git vim tree
      echo 'export PATH=$PATH:/opt/puppetlabs/puppet/bin' >> /root/.bashrc
    }
  end
  config.vm.define 'agent' do |agent|
    agent.vm.hostname = 'agent.localdomain'
    agent.vm.network "private_network", type: "dhcp"
    agent.vm.provision "shell", inline: %Q{
      cp -r /vagrant/.vim* /root
      rpm -Uvh https://yum.puppetlabs.com/puppetlabs-release-pc1-el-7.noarch.rpm
      yum -y install git vim
      echo 'export PATH=$PATH:/opt/puppetlabs/puppet/bin' >> /root/.bashrc
    }
  end
end
```
    - Download vagrant_2.2.19_x86_64.deb (works for Ubuntu18) and  run `vagrant up --provide virtualbox`
      - `agent` was not created. Try again `vagrant up agent`
      - To halt, `vagrant halt`
      - To remove VM, `vagrant destroy`
    - `vagrant status`
    - `vagrant ssh puppet`
      - `sudo yum install puppet-server`
      - Installed at /opt/puppetlabs
    - `vagrant ssh agent`
      - `sudo yum install puppet-agent`
    

4. Puppet Resources
  - Defining state the idempotent way
    - yum install php5
    - Or may configure apt command for Ubuntu
  - Defining state in puppet
```
  package { 'php5' : ensure -> installed,}
```
  - Resoruce types
    - Package: `package { 'php5' : ensure -> installed,}`
    - User: `user { 'bob' : ensure -> present,}`
    - Group
    - Service
    - File
  - Resource Abstraction Layer (RAL)
    - Provides abstgraction from the implementation details
    - We declare WHAT, and the RAL decides HOW
    - The Point where Puppet interacts with your OS
  - Resource types and providers
    - User: adduser, useradd
    - Package: yum
  - Use `puppet resource` to interact with RAL
    - Can create or change attribute (idempotency)
  - Attributes
    - Resource types have a number of configurable attributes
    - UID, GID, home directory, shell
    - `puppet describe` shows what attributes a type supports
    - `puppet describe --list`

5. The Puppet Language
  - Features
    - A declarative syntax
    - Readable and easy to understand
    - Infrastructure as code
  - Resource declarations
    - The resource type that we are managing
    - The identifying name of the resource
    - Attributes that we want to manage
    ```
    type { 'title':
      attribute => 'value',
    }
    ```
  - Manifests
    - Resource declarations are written in manifest files with .pp extension
    - `puppet apply` command
    - Resources must be unique, and you cannot define two states for the same resource
  - Classes
    - Re-usable configuration models
    - Singleton
    - Model configuration by grouping together resources
    - Apply the class to instantiate the configuration
    ```
    class sysadmins {
      group { 'sysadmins':
        ensure => present,
      }
      user { 'bob':
        ensure => present,
        uid => '9999',
        groups => 'sysadmins',
        }
    } 
    ```   
    - This is definition, not applying
    - Use `include sysadmins` to apply
  - Modules
    - Classes are contained within modules
    - Contains:
      - File serving
      - Puppet extensions
    - Specific file system layout under modulepath
      - `puppet config print` to see the modulepath
      - `/etc/puppetlabs/code/environments/production/modules
  - Running a standalone puppet code: `puppet apply -e <code.pp>`

6. Puppet Server and Agent
- Puppet agent
  - rUns as a daemon checking periodically
  - -o : run once and exit
  - -n: run in the foreground
  - -v: verbose mode
  - -t: implies -onv
  - --noop: compare catalog but doesn't apply change
- Authentication
  - Agents connects to server over authenticated SSL
  - Retrieves and applies configuration
- SSL
  - Puppet server is a certificate authority (CA)
  - The agent generates an SSL signing request (CSR) and sends it to the server
  - Server signs and returns to the cert to the agent
  - Puppet uses the certname to identify hosts
  - Agent verifies the server SSL cert contains the certname
  - `puppet cert` to manage certificates
- Common SSL issues
  - Clocks out of sync. Running NTP is advised
  - Certname mismatch b/w servers and agetns
  - Agents has been rebuilt
  - Removing certs
    - `puppet cert clean agent.localdomain` in server
    - `puppet config print ssldir` then remove the folder in agent
    - Then regenerate certs
- Facter
  - Facter gathers a wide variety of information about the agent node
  - Puppet can use these facts to determine how to configure resources
  - Easily extendable to add your own facts
  - `facter` to view them
    - `facter operatingsystem`
    - `facter osfamily`
- Catalog
  - Contains all managed resources and the state they must be in
  - Puppet agent uses the RAL to compare running state to catalog
  - Changes are enforced where drift is detected
- Classification
  - When an authenticated agent checks in, it is classified
  - Puppet identifies which classes must be applied
  - Catalog is compiled
- How to classify
  - External node classifier (ENC)
    - Enterprise Console: commercial one
    - Foreman: for OSS
  - Manifest file (site.pp)
- Node definitions: uses REGEX

## Section 3: Module 2: The Puppet Language

7. More Resources
- Package: ensure absent, installed, latest or a specific version
- Service: ensure can be `running` or `stopped`
- Notify: `notify {'Hello': }`
- EXEC
  - Execute any arbitrary command: `exec { 'configure app' : path => '/bin:/usr/bin', command=> 'configure.sh', creates => '/bin/myapp', }`
    - If `/bin/myapp` doesn't exist, puppet will run myapp
    - We may use `unless` keyword instead of `creates`
- NAMEVAR
  - Puppet resources must be unique
  - Normally the title of the resource is used as the unique identifier
  - Each resource type also has one or more attributes that are **namevars**
  - Resource titles and **namevars** must be unique
```
package { 'web server':
   ensure => installed,
   name   => 'httpd',
}
```
  - Yum command will use the value of `name`
  - Notice will show `web server` as resource title
- Puppet describe
  - `puppet describe --list`: list of available resources
  - `puppet describe <resource type>`: configurable attributes of a resource type

8. File Saving
- File resource type
  - Manage files, directories and symlinks
  - `ensure` can be file, symlink or directories
  - owner, group, and mode can be modified
- Managing file contents
  - Statically or dynamically
  - `source` specifies a location on the puppet server to serve the file statically
```
file { '/etc/httpd/httpd.conf': 
  ensure => file,
  source => 'puppet:///modules/apache/httpd.conf',
  owner  => 'root',
  group  => 'root',
  mode   => '0644',
  }
```
  - `content` specifies a string value to populate the file dynamically
```
file { '/etc/motd': 
  ensure  => file,
  content => 'Welcome to my system',
  owner   => 'root',
  group   => 'root',
  mode    => '0644',
  }
```
- Static files
  - The URI is made up of a hostname, a mountpoint, and a path: `puppet://<hostname>/<mountpoint>/<path>`
  - If a host is not specified, puppet retrieves file from server currently in use for the session: `puppet:///<mountpoint>/<path>`
  - Mountpoint corresponds to an entry in `fileserver.conf`
  - Modules mountpoint is a special internal mountpoint. The server will search in a predetermined location in the `modulepath`: `puppet:///modules/<path>`
    - The path used for the modules mountpoint is `<modulename>/<filename>`
    - The file name is relative from the `files` folder in: `<modulename>/files/<filename>`
    - Ex) `puppet:///modules/apache/httpd.conf`
      - Actual file structure is `apache/files/httpd.conf`
```
class motd {
  file {'/etc/motd':
  ensure => file,
  source => 'puppet:///modules/motd/motd.txt',
  owner  => 'root',
  group  => 'root',
  mode   => '0644',
  replace => false,
  }
}
```
  - As `replace` is false, it will not change the file if it exists

9. Relationships
- Resource ordering
  - Resources are read in the order they are written
  - That doesn't always mean that's the order they will be applied in
  - Manifest ordering is now default but explicit relationships are cleare especially with resources across multiple classes
  - The ordering option defaults to manifest is later version of Puppet but could be title-hash or random
    - `puppet apply my.pp --ordering title-hash`
    - `puppet apply my.pp --ordering random`
- Referencing resource
  - A resource reference is a pointer to a resource declaration: `Package[ 'httpd' ]`
  - Can configure dependency with `require`
    - Resource must be declared in puppet explicitly. Existing resource without declaration will not work
  ```
  package { 'httpd':
    ensure => installed,
  }
  service { 'httpd': 
    ensure => running,
    require => Package['httpd'],
  }
  ```
  - Or using `before` attribute:
  ```
  package { 'httpd':
    ensure => installed,
    before => Service['httpd'],
  }
  service { 'httpd': 
    ensure => running,
  }
  ```
- Triggering events
  - Some resources need to take an action based on an event occurring in a different resource
  - For example when a configuration file is changed we want to restrt the service
  - Some resource types are **refreshable**, meaning they take an acition when they receive a refresh event
  - Ex) We want to restart httpd when apach configuration file is pupdated
    - Use `subcribe` to send an event notification
  ```
  file { '/etc/httpd/httpd.conf':
    ensure => file,
    source => 'puppet:///modules/apache/httpd.conf',
  }
  service { 'httpd':
    ensure => running,
    subscribe => File ['/etc/httpd/httpd.conf'],
  }
  ```
    - Or use `notify` 
  ```
  file { '/etc/httpd/httpd.conf':
    ensure => file,
    source => 'puppet:///modules/apache/httpd.conf',
    notify => Service ['httpd'],
  }
  service { 'httpd':
    ensure => running,
  }
  ```
  - The `exec` resource is also refreshable
  ```
  service {'tinpot':
    ensure => running,
    enable => true,
    notify => Exec['clean tinpot cache'].
  }
  exec { 'clean tinpot cache': 
    path        => '/opt/tinpot/bin',
    command     => 'tinpot --cleancache',
    refreshonly => true,
  }
  ```
- Implied dependencies
  - `user` resource type auto-requires any groups specified
  ```
  user { 'bob' :
    ensure => present,
    groups => 'sysadmins',
  }
  group { 'sysadmins':
    ensure => present,
  }
  ```
  - Also `file` resource type vs `owner` attribute
- Resource chaining
  - A short hand syntax for expressing relationships by referencing the resources and chaining them together
  - `Package['httpd'] -> File['/etc/httpd/httpd.conf']`
```
package { 'httpd':
  ensure => installed,
} ->
file { '/etc/httpd/httpd.conf':
  ensure => file,
  source => 'puppet:///modules/apache/httpd.conf'
}
```
  - `->` left before right
  - `<-` right before left
  - `~>` left refreshes right (this is tilde, not dash)
  - `<~` right refreshes left
- `Package['httpd'] -> File['/etc/httpd/httpd.conf'] ~> Service['httpd']` is equivalent to `Service['httpd'] <~ File['etc/httpd/httpd.conf'] <- Package['httpd']`

10. Interactive exercise: Package/File/Service

11. Variables

12. Conditionals

13. Data Types

14. Functions

15. Templates

16. Parameterized Classes

17. Defined Resource Types

18. Advanced Resource Declarations in Puppet4
