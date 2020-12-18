# Learn Ruby on Rails from Scratch
- ? Screen capture uses too high resolution. Hard to read what the lecturer does
- Not organized well. Hard to understand what the lecture wants to deliver
- Not recommended

## Section 4. Install Ruby on Ubuntu
- Ignore lecture and use followings:
  - sudo apt install ruby
  - sudo apt install ruby-dev
  - sudo apt install -y nodejs
  - sudo gem update --system
  - gem -v
  - sudo gem install rails
  - rails -v

### Another installation guide
- ref:https://www.howtoforge.com/tutorial/how-to-install-ruby-on-rails-on-ubuntu-1804-lts/
- gpg --keyserver hkp://keys.gnupg.net --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3 \
7D2BAF1CF37B13E2069D6956105BD0E739499BDB
- curl -sSL https://get.rvm.io | bash -s stable --ruby
- source /usr/local/rvm/scripts/rvm
- rvm get stable --autolibs=enable
- usermod -a -G rvm root
- rvm list known
- rvm install ruby-X.X.X
- rvm --default use ruby-2.5.1
- ruby -v
- gem update --system
- gem -v
- gem install rails -v 5.2.0
- rails -v
### Update of Nodejs
- curl -sL https://deb.nodesource.com/setup_11.x | sudo -E bash -
- sudo apt install -y nodejs
### Update of Yarn
- curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
- echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
- sudo apt update
- sudo apt install yarn
### PostgreSQL
- sudo apt install postgresql postgresql-contrib libpq-dev -y
- sudo gem install pg # ruby gem for postgresql


## Section 5. A sample app
- `rails new ~/TEMP/myruby`
  - Enter the entire path
  - Will copy/install template
- `cd myruby`
- myruby folder
  - app : views, modles, controllers, ...
  - config : database related
  - db: database migration
  - public: css and images
- `rails server`
  - Goto web browser and localhost:3000
- `rails generate controller home index`
- Edit config/routes.rb
  - Insert `root :to => 'home#index'` then the main page refers app/views/home/index.html.erb

## Section 10. IRB
- irb<main>:line_number:layer_number>
- print("hello world") ; puts "hello world"
- nil means no error
- For floating number, bothf of dividend and divisor must be floats

## Section 11. Methods and Conditions
```
def func(x)
return x*x
end
```
- Depending on the input, the number of layer changes
- To see the methods of an object: "Hello".methods
  - Equivalent to dir("hello") in python

## Section 12. Array hash and loops
- Arrays use []
  - `x = [1,3,7]`
- Array hash
  - `user = {'firstname'=> 'John',  'lastname'=>'Smith' }`
  - `user = [ {'firstname'=> 'John',  'lastname'=>'Smith' }, { 'firstname'=>'Brad', 'lastname'=>'Trav'} ]`
  - Corresponds to dictionary in Python
- Loops
  - `5.times do puts "Hello world" end`
    - New line after Hello world
  - `5.times { print("Hello world") }`
    - No new line. Texts will be concatenated
  - `3.upto(10) {puts "Hello world"}`
    - 8 times repetitions
  - `3.downto(1) {puts "Hello world"}`
    - 3 times repetitions
  - atoms = [ "Hyd", "Hel", "Li"]; atoms.each{ |x| puts x}
```
irb(main):003:0> while x < limit
irb(main):004:1> puts x
irb(main):005:1> x += 1
irb(main):006:1> end
```

## Section 13. Objects and classes
- Defining class components
```
irb(main):007:0> class Car
irb(main):008:1> attr_accessor :make, :model, :color
irb(main):009:1> end
```
  - car1 = Car.new # makes a new class object
  - car1.make = 'Honda'
- Defining class functions
```
irb(main):020:0> class Car
irb(main):021:1> def drive
irb(main):022:2> print 'Driving'
irb(main):023:2> end
irb(main):024:1> end
```
  - car1 = Car.new; car1.drive
- Inheriting classes
```
irb(main):001:0> class Vehicle
irb(main):002:1> attr_accessor :make, :color, :year
irb(main):003:1> end
=> nil
irb(main):004:0> class Car < Vehicle
irb(main):005:1> attr_accessor :fourwheel
irb(main):006:1> end
```

## Section 17. Generating Controllers and Views
- At myruby/, rails generate controller posts
  - This will generate myruby/app/views/posts folder
  - Edit myruby/app/controllers/posts_controller.rb
```
class PostsController < ApplicationController
  def index

  end
end
```  
  - Provide myruby/app/views/posts/index.html.erb
```
<h1> Blog Posts </h1>
<h3> Sample Post </h3>
<p> This is a sample text for new Ruby app </p>
<hr>
<h3> Sample Post 2 </h3>
<p> This is a 2nd sample text for new Ruby app </p>
```
  - Update app/config/routes.rb
```
Rails.application.routes.draw do
  get 'home/index'
    resources  :posts
  root :to => 'home#index'
end
```
  - Now load 127.0.0.1:3000/posts

## Section 18. Action Controllers & Routes
- At myruby folder, `rake routes`
  - Shows Prefix, Verb, URI Pattern
- Make app/views/posts/edit.html.erb, show.html.erb
  - Provide html grammar within
- In app/controllers/posts_controller.rb, provide functions of edit, show such as
```
class PostsController < ApplicationController
  def index
    @content_first = 'This is a content from posts_controller'
    @content_second = 'This is a function from posts_controller'
  end
  def edit
  end
  def show
  end
```
- Now load `127.0.0.1:3000/posts/edit`
- In app/views/posts/index.html.erb, contents from app/controllers/posts_controller.rb can be used by injecting `<p><%=@content_first %> </p>`

## Section 19. Database planning and model creation
- sudo gem install pg # install postgresql gem module
- Update config/database.yml for postgresql
  - Ref: https://gist.github.com/jwo/4512764
- rails g model post title:string body:text category_id:integer author_id:integer
- rake db:migrate

### start from scratch again
- rails new myrubyblog --database=postgresql --webpack=react
  - Will install gems. Supply sudo passwd when requested
- rails g model post title:string body:text category_id:integer author_id:integer
- rake db:migrate


 sudo -u postgres psql
[sudo] password for hpjeon:
psql (12.5 (Ubuntu 12.5-1.pgdg18.04+1), server 10.15 (Ubuntu 10.15-1.pgdg18.04+1))
Type "help" for help.
```
sudo -u postgres psql
postgres=# \du  # list of users
postgres=# ALTER ROLE postgres
postgres-# WITH PASSWORD 'XXXXX';
## make sure ; is added. After update, "ALTER ROLE" must be prompted in CLI
postgres-# \q
```
- To login from CLI, psql -U postgres -h localhost
  - This will ask passwd
 export DATABASE_URL="postgres://postgres:XXXXXX@localhost/myrubyblog"
hpjeon@hakune:~/TEMP/myrubyblog$ !ra

## digital ocean.com
- Ref: https://www.digitalocean.com/community/tutorials/how-to-use-postgresql-with-your-ruby-on-rails-application-on-ubuntu-18-04
- Create ROR application with postgresql
```
rails new utau -d=postgresql
cd utau
```
- Edit config/database.yml and below default:, add username and passwd
```
default: &default
  adapter: postgresql
  encoding: unicode
  host: localhost
  pool: 5
  username: postgres
  password: XXXXX
  port: 5432
```
- rails db:create
  - From pgadmin4, confirm that utau_development and utau_test databases are created
- rails server --binding=localhost
  - From web-browser, enter `localhost:3000`

## medium.com class
- https://medium.com/@frouster/ruby-on-rails-postgresql-f1b037924bdf
