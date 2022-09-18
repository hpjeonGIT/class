## Master Puppet for DevOps Success
- Instructor: Craig Dunn

## Section 1: Introduction

1. Introduction

## Section 2: Module 1: Getting Started with Puppet

2. Why Puppet
  - Automation general
    - Cobbler/Kickstart
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
```ruby
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
    ```ruby
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
    ```ruby
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
```ruby
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
```ruby
file { '/etc/httpd/httpd.conf': 
  ensure => file,
  source => 'puppet:///modules/apache/httpd.conf',
  owner  => 'root',
  group  => 'root',
  mode   => '0644',
  }
```
  - `content` specifies a string value to populate the file dynamically
```ruby
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
```ruby
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
  ```ruby
  package { 'httpd':
    ensure => installed,
  }
  service { 'httpd': 
    ensure => running,
    require => Package['httpd'],
  }
  ```
  - Or using `before` attribute:
  ```ruby
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
  ```ruby
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
  ```ruby
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
  ```ruby
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
  ```ruby
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
```ruby
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
- An exercise of apache module
  - Manage the httpd package and service
  - Manage the default httpd.conf file
  - Ensure the document root exists
```bash
cd /etc/puppetlabs/code/environments/production/modules
mkdir apache
cd apache
mkdir manifests
mkdir files
cd files
cp /root/httpd_minimal.conf .
cd ../manifests
vim init.pp ## see below
cd /etc/puppetlabs/code/environment/production/manifests
vim site.pp ## see below
```
- init.pp
```ruby
class apache {
  package { 'httpd':
    ensure => installed,
  }
  file { '/etc/httpd/conf/httpd.conf':
    ensure => file,
    source => 'puppet:///modules/apache/httpd_minimal.conf',
    require => Package['httpd'],
  }
  service { 'httpd':
    ensure => running,
    enable => true,
    subscribe => File ['/etc/httpd/conf/httpd.conf'],
  }
  file { '/var/www/html':
    ensure => directory,
  }
}
```
  - Make sure the chain of dependency
- site.pp
```ruby
node "agent.localdomain" {
  include motd
  include apache
}
```
- Now run `puppet agent -t` from the agent node
  - Check through `systemctl status httpd`

11. Variables
- Prefixed with `$`
- Assigned with `=`
- Must begin with a lower case letter or underscore
  - `$pkgname = 'apache'`
- Once declared, cannot be modified or re-declared
- Can be used as a resource titles/attribute values
  - `package { $pkgname: .... }`
- Strings in Puppet must be quoted
- Single quotes for static content
- **Double quotes** for interpolated content  
- When interpolating a variable into a string, the variable must be in brackets
  - Use brackets within double quotes always
```ruby
$prefix = 'README'
$suffix = 'txt'
$filename = "${prefix}.${suffix}"
```
- Rewriting with variables
```ruby
class apache {
  $package_name = 'httpd'
  $service_name = 'httpd'
  $config_file = '/etc/httpd/conf/httpd.conf'
  package { $package_name:
    ensure => installed,
  }
  file { $config_file:
    ensure => file,
    source => 'puppet:///modules/apache/httpd_minimal.conf',
    require => Package[$package_name],
  }
  service { $service_name:
    ensure => running,
    enable => true,
    subscribe => File [$config_file],
  }
}
```
- Arrays
  - Declared inside square brackets
  - Can use an array in the resource title, creating multiple resources
  - Some resource types can use an array
```ruby
$users = ['bob', 'susan', 'peter']
user { $users: 
  ensure=> present,
}
file {'/etc/app.conf':
  ensure => file,
  require => Package['httpd', 'acmeapp'],
}
file {'/etc/web.conf':
  ensure => file,
  require => [Package['httpd'], Service['acmep']],
}
```  
- Hashes (dictionary)
  - Using brackets {}
  - Key valeus are separated by a hashrocket (=>)
    - Similar to Ruby
```ruby
$uids = { 
  'bob'   => '9999',
  'susan' => '9998',
  'peter' => '9997',
}
$uid_susan = $uids['susan']
```
- Scope
  - If a variable is not found in the current scope, the next scope is searched
  - Scopes are namespaced with `::`
- Facts revisited
  - Agent facts are sent to the server and available in a hash called `$::facts`
  - We can use the `facter` command to view facts on the CLI
  - Facts are also top level variables, but the `$::facts` has is recommended
  - Ex) `$::facts['os']['family']`
- Trusted facts
  - From Agents SSL certificate
  - Stoped in a top level hash called `$::trusted`
  - For security, instead of `$::facts['certname']`, `$::trusted['certname']` must be used

12. Conditionals
- Assignment conditionals
  - Selectors
    - Assign data based on evaluation
    - Not control the flow of the code
  ```ruby
  $package_name = $::facts['os']['family'] ? {
    'Debian' => 'apache2',
    'Redhat' => 'httpd',
    'Solaris' => 'CSWApache2',
    default => 'httpd',
  }
  ```
- Flow conditionals
  - Case statements
    - Executes a block of code depending on the evaluation
  ```ruby
  case $::facts['os']['family'] {
    'Redhat': {
      include yum
    }
    'Debian': {
      include apt
    }
    default: {
      fail ('Unknown OS')
    }
  }
  ```
  - if/else blocks
    - `==`: equal
    - `!=`: not equal
    - `<`: less than
    - `<=`: less than or equal to
    - `>`: greater than
    - `>=`: greater than or equal to
    - `=~`: match regular expression
  ```ruby
  if $install_package {
    package { $packagename:
      ensure=>installed,
    }
  }
  if $environment == 'dev' {
    include devutils
  }
  ```
- Regular expression
  - Surround using slashes as `/.../` 
```ruby
case $::trusted['certname'] {
  /.*\.uk\..*/: {
    include users::london
  }
  /.*\.es\..*/: {
    incluide users::madrid    
  }
  default: {
    fail ('Not in the UK')
  }
}
$country = $domain ? {
  /.*\.uk$/ => 'United Kingdom',
  /.*\.es$/ => 'Spain',
  default   => 'Other palce',
}
```
- Case sensitivity
  - By default, puppet will use **case insensitivity**
```ruby
$pacakge_manager = $::facts['os']['family'] ? {
  'redhat' => 'yum',
  'debian' => 'opt',
  default => 'unknown',
}
```
  - 'redhat' or 'RedHat' will work
  - To enforce case sensitivity, use `/.../` such as `/RedHat/ => 'yum'`

13. Data Types
- Datatype begins with a Capital letter like `String`
- String
  - `String[4,8]`: strings of 4-8 characters
- Integer/Float/Numeric
  - `Integer[5,10]`: integer b/w 5-10
  - `Float[1.9]`: min value as of 1.9
  - `Numeric` doesn't support parameters of min/max
- Array
  - `Array[String]`
  - `Array[Any,5]`: Any datatype with 5 array size
- Hash
  - `Hash[String, Integer]`
  - `Hash[String, Integer, 5,10]`
- `Regexp`
- `Undef`
- `Variant[String, Intger]`
- `Enum['yes','no']`
- `Optional[String]`
- Datatype comparison using `=~`
  - Output will be true or false
```ruby
$username = 'micky'
unless $username =~ String[4,8] {
  fail('Invalid username')
}  
```
  - Checks a string type with 4-8 characters

14. Functions
- Server side methods
  - Functions can be written in Ruby or Puppet DSL
  - Executed server side during catalog compilation
  - Functions either return data to assign to a variable or perform an action with no return value
- Calling functions
  - Prefixed syntax: funciton(arg,arg)  
    - `notice("Hello world")`
  - Chained syntax notation: arg.function(arg,arg)
    - `"Hello World".notice`
- Functions can be found at 3 levels
  - Global function: `function`
  - Environment level function: `environment::function`
  - Module funciton: `modulename::funciton`
- Lambdas (code blocks)  
  - Anonymous functions
  - Prefixed syntax: `function(arg,arg) |param,param| { }`
  - Chained syntax: `argument.function(arg,arg) |param,param| { }`
```ruby
$user = with("susan", "susan@example.com") |$u, $e| {
  { "user_name" => $u,
    "email"     => $e,
    }
}
```  
- Loops and iterators
  - Iterates over hashes or arrays  
```ruby
$vhosts = ['acme.com','examle.com', 'abc.org']
$vhosts.each | $v | {
  file { ["/var/sites/${v}", "/var/log/vhots/${v}" ]:
    ensure => directory,
  }
}
```
- Data validation
  - Use datatype explicitly
  - Ex) `$vhosts.each | String $hostname, Integer $port | {...}`
```ruby
$users = ['kate','susan', 'fred']
$users.each | String $u| {
  user {$u:
    ensure     => present,
    managehome => true,
  }
  file { "/home/${u}/.bashrc":
    ensure => file,
    owner  => $u,
    group  => $u,
    content => 'export PATH=$PATH:/opt/puppetlabs/puppet/bin',
  }
}
```
- Writing functions
  - Puppet supports three types of functions
    - Legacy Ruby API (will be deprecated)
      - Global name space. Very dangerous
    - Modern Ruby API
      - Scope is defined through namespace
    - Puppet DSL functions
- Function demo
  - Given a hostname of xxx-{p,d,q}-yyy return the hosts's environment by matching
    - p = Production
    - d = Development
    - q = QA
  ```
  cd /etc/puppetlabs/code/environments/production/modules
  mkdir hostname
  cd hostname
  mkdir manifests
  mkdir functions
  cd functions
  vim environment.pp
  cd ../manifests
  vim init.pp
  puppet apply -e 'include hostname'
  ```
  - environment.pp
  ```ruby
  function hostname::environment(String $host) >> String {
    $env_name = $host ? {
      /[^\-]+-p-[^\-]+/ => 'Production',
      /[^\-]+-d-[^\-]+/ => 'Development',
      /[^\-]+-q-[^\-]+/ => 'QA',
    }
    $env_name
  }
  ```
  - init.pp
  ```ruby
  class hostname {
    $server_name = 'oradb-p-001'
    $environment_name = hostname::environment($server_name)
    notify { "My environment is ${environment_name}": }
  }
  ```
- Module layout
  - files
  - functions -> Puppet DSL functions
  - lib
    - puppet
      - functions -> Ruby functions
  - manifests

15. Templates
- Using puppet templates
  - We embed code into static content to create templates
  - Templates are normally used for serving dynamic file contents
  - Puppet supports 2 template formats
    - EPP (Embedded Puppet), native puppet DSL templates
    - ERB (Embedded Ruby), Legacy Ruby Templates
  - EPP templates are called using the in-built epp function
  - Tempaltes are served from the templates folder directly under the module root: `<modulepath>/<module name>/templates/name.epp`
- Calling puppet templates
  - The epp function is used to render a template and return the content
  - The first argument is the location of the template as `<modulename>/<file>`
  - The second argument is a hash of parameters to pass to the template
    - Key/values are defined in this hash (dictionary)
  ```ruby
  $params = { 'role' => 'database', 'server_name' => 'oracle01' }
  $output = epp('mymodule/welcome.epp', $params)
  ```
- Writing puppet templates
  - Templates are static content with embedded dynamic tags surrounded by `<%...%>`
  - There are 3 types of tags
    - `<% | ... | %>`: parameter tag
    - `<%  ...  %>`: functional tag
    - `<%= ...  %>`: expression substitution tag
  ```ruby
  <% | String $role, String $servr_name |%>
  Welcome to <%= $server_name %>
  This machine is a <%= $role %> server
  ```
- Blank lines
  - Ending with `%>` yields new line
  - To avoid the new line, end with `-%>`

16. Parameterized Classes
- Puppet classes must be designed to be re-usable, sharable components
- We can use class parameters to make instantiation of a class customizable
- Implements class like a function, feeding arguments
  - Using the resource declaration syntax, we can pass parameters to a class the same way as resource attributes
    - Default values are allowed
    - Datatype might be enforced
  ```ruby
  class apache ( 
    String $version, 
    String $docroot, 
    String $bindaddress, 
    Integer $port=80) {...}
  ...
  class { 'apache' :
    version => '2.2.3',
    docroot => '/sites/default',
    bindaddress => '10.0.1.15',
    port => 80,
  }
  ```

17. Defined Resource Types
- Introduction
  - Similar to classes, but provides a configuration model that can be instantiated multiple times
  - When you need to group together several resources into a repeatable template
- Example
  - Apache resource regarding conf file/doc root/index.html  
    - Couple them as vhost
- Writing defined resource types
  - Named as `<module>::<name>`
  - They should be placed in a manifest file that corresponds with the name
  - A defined resource of `apache::vhost`is defined in `<modulepath>/apache/manifests/vhost.pp`
  - Defined resources are written using the `define` keyword
  - The syntax is almost identical to a writing a class
```ruby
define apache::vhost (
  String $docroot = "/var/www/${name}",
  Integer $port,
) {
  file { $docroot:
    ensure => directory,
  }
  file { "/etc/httpd/conf.d/${name}.conf":
    ensure => file,
    content => epp('apache/vhost.epp', { "port" => $port }),
  }
  file { "${docroot}/index.html":
    ensure => file,
  }
}
```
- Declaring a defined resource
  - Defined resources are declared when the same syntax as regular resources
  - The resource attributes are passed to the parameters of the defined resource type
  - The resource title is passed to the `$name` variable above
```ruby
apache::vhost { "acme.com":
  port => 80,
}
```
- Q: Is `$name` pre-defined position holder for the resource title?
```ruby
apache::vhost { "example.com":
  port => 81,
  docroot => '/sites/example.com',
}
apache::vhost { "foo.com":
  port => 80,
  docroot => '/sites/foo.com',
}
apache::vhost { "acme.com":
  port => 80,
  docroot => '/sites/acme.com',
}
```

18. Advanced Resource Declarations in Puppet4
- Best practice
- Resource grouping
  - Multiple resource of the same type can be long to write
  - We can declare multiple resources within the same resource declaration block
  - 3 resources separately
  ```ruby
  file {
    '/etc/foo':
    ensure => file,
    source => 'puppet:///mymodule/foo',
  }
  file {
    '/etc/bar':
    ensure => file,
    source => 'puppet:///mymodule/bar',
  }
  file {
    '/etc/tango':
    ensure => file,
    source => 'puppet:///mymodule/tango',
  }
  ```
  - Grouping: can save some lines. Use semicolon `;` to separate resources
  ```ruby
  file {
    '/etc/foo':
      ensure => file,
      source => 'puppet:///mymodule/foo';
    '/etc/bar':
      ensure => file,
      source => 'puppet:///mymodule/bar';
    '/etc/tango':
      ensure => file,
      source => 'puppet:///mymodule/tango';
  }
  ```
- Resource defaults
  - When declaring lots of resources of the same type with identical attributes, we can reduce the amount of code by using resource defaults
  - Puppet supports two types of resource defaults
    - Reference syntax
    - Resource declaration syntax
```ruby
File {
  owner => 'root',
  group => 'root',
  mode  => '0777',
}
file {
  '/etc/foo':
    ensure => file;
  '/etc/bar':
    ensure => file;
  '/etc/tango':
    ensure => file;
}
```
- In puppet4, we use `default` keyword
```ruby
file {
  default:
    owner => 'root',
    group => 'root',
    mode  => '0777';
  '/etc/foo':
    ensure => file;
  '/etc/bar':
    ensure => file;
  '/etc/tango':
    ensure => file;
}
```
- Dynamic attributes
  - Puppet supports the ability to define resource attributes dynamically from a hash
  - To pass dynamic resources to a declaration we can use the special `*` attribute
```ruby
$attrs = {
  'owner' => 'root',
  'group' => 'root',
  'mode'  => '0644',
}
file { '/tmp/foo':
  ensure => file,
  *      => $attrs,  
}
```
  - `*` can be used in default attributes as well

## Section 4: Conclusion

19. Next Steps
  - Hiera
