# Learn Ruby on Rails from Scratch

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

## Section 12. Array and loops
- Arrays use []
  - `x = [1,3,7]`
