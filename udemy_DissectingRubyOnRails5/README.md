## Dissecting Ruby on Rails 5
- By Jordan Hudgens

## 32. Creating a Rails application
- `rails new DevcampPortfolio -T --database=postgresql`
  - `-T` will not install Test Framework
- Edit config/database.yml and add username, password, and host
```
username: postgres
password: XXXXX
host: localhost
```
- `rails db:create`
- `rails db:migrate`
- `rails server` then open localhost:3000 in a web-browser

## 33. Generating a Blog using Rails Scaffolds
- `rails g scaffold Blog title:string body:text`
- Scaffold system will produce everything for Blog
- Check db/migrate/20201219165315_create_blogs.rb
- `rails db:migrate`
  - Will create a new table
  - `db/schema.rb` is updated. Don't change this file manually
- `config/routes.rb` will configure route information
- Open `http://localhost:3000/blogs` and try enter a new blog
- A prototype of Blog system is available now

## 34. How controller works
- app/assets/stylesheets/scaffolds.scss has html configuration of the web page. In this class, we delete this file
- app/controllers/blogs_controller.rb: Actual web-page behavior
  ```
  def index
    @blogs = Blog.all
  end
  ```
  - If `Blog.limit(1)` is used, only the first data will be shown
  - `def show` is empty as `before_action :set_blog, only: [:show, :edit, :update, :destroy]` is applied in the head. `:set_blog` is defined below as `@blog = Blog.find(params[:id])`. This function is executed at show, edit, update, and destroy function while this practice reduces duplication of the same function calls

## 35. Create, Update, and Destroy
- Each function in app/controllers/blogs_controller.rb is coupled with app/viewers/blogs/*.erb file such as index.html.erb or new.html.erb
- def new vs def create
  - new is used for new blog button - asking new data input
  - create is to create new table row
- `blog_params` is defined in the bottom of blogs_controller.rb as a method

## 36. Routing
- config/routes.rb
- Command `rake routes` shows all of the routes
```
                              Prefix Verb   URI Pattern                                                                              Controller#Action
                                blogs GET    /blogs(.:format)                                                                         blogs#index
                                      POST   /blogs(.:format)                                                                         blogs#create
                             new_blog GET    /blogs/new(.:format)                                                                     blogs#new
                            edit_blog GET    /blogs/:id/edit(.:format)                                                                blogs#edit
                                 blog GET    /blogs/:id(.:format)                                                                     blogs#show
                                      PATCH  /blogs/:id(.:format)                                                                     blogs#update
                                      PUT    /blogs/:id(.:format)                                                                     blogs#update
                                      DELETE /blogs/:id(.:format)                                                                     blogs#destroy
        rails_postmark_inbound_emails POST   /rails/action_mailbox/postmark/inbound_emails(.:format)                                  action_mailbox/ingresses/postmark/inbound_emails#create
           rails_relay_inbound_emails POST   /rails/action_mailbox/relay/inbound_emails(.:format)                                     action_mailbox/ingresses/relay/inbound_emails#create
...
```
- `blogs#index`, `blogs#create`, ... are configured in the routes
- `:id` means that URL expects a kind of id number

## 37. Rails file system
- Convention over configuration
  - Names of files/variables are coupled each other

## 39. Rail generation test
- `rails -h` shows the list of options to use
- `rails new my_app -B` # -B will skill bundle install, and this will be fast
- `rails new my_api --api -T` # will generate without test framework
  - Produces json related components only

## 50. Rails generator
- Controller generator
- Model generator
- Resource generator

## 51. Controller generator
- Scaffold is great but may generate too many codes
- `rails g controller Pages home about contact`
  - Not using scaffold
  - app/controllers/pages_controller.rb is made while function definitions are empty
  - Coupled with config/routes.rb
- MVC: Model - View - Controller
  - Model: app/models/blog.rb
  - View: app/views/pages/home.html.erb
  - Controller: app/controllers/pages_controller.rb

## 53. Model generator
- `rails g model Skill title:string percent_utilized:integer`
  - app/models/skill.rb and db/migrate/*_create-skills.rb are produced
- `rails db:migrate`
  - Updates the database. db/schema.rb is updated
- `rails c` # rails console. Can connect to databa
  - `Skill.create!(title: "Rails", percent_utilized: 75)` # add a new row to Skill table
  - `Skill.all` # shows all rows of Skill table
- Inject `@skills = Skill.all` to `def home` of app/controller/pages_controller.rb
- Update app/view/pages/home.html.erb as
```
<h1>Homepage</h1>
<p>Find me in app/views/pages/home.html.erb</p>
<h1>Blog Posts</h1>
<%= @posts.inspect %>
<h1>Blog Posts</h1>
<%= @skills.inspect %>
```
- Now web-page (http://localhost:3000/pages/home) shows the updated Skill table

## 54. Resource generator
- `rails g resource Portfolio title:string subtitle:string body:text main_image:text thumb_image:text`
  - For `main_mage:text`, the text is the link of an image
  - Generates app/models/portfolio.rb, app/controllers/portfolios_controller.rb, app/views/portfolios, ...
- `rails db:migrate`
  - Check db/schema.rb is updated

## 55-56. Deep Dive: Rails Generators and customization
- rails new GeneratorApp -T --database=postgresql
- cd GeneratorApp
  - Edit config/database.yml for username and password
- rails db:create
- rails g scaffold Post title:string body:text
- rails db:migrate
- Make a new folder at lib/templates/erb/scaffold/index.html.erb

## 58. Seeds file
- db/seeds.rb
```
10.times do |blog|
  Blog.create!(
    title: "My Blog Post #{blog}",
    body: "some random texts here"
  )
end
5.times do |skill|
  Skill.create!(
    title:"Rails #{skill}",
    percent_utilized:15
  )
end
9.times do |portfolio_item|
  Portfolio.create!(
    title: "Portfolio title: #{portfolio_item}",
    subtitle:"My great service",
    body: "Some random texts",
    main_image: "http://placehold.it/500x300",
    thumb_image: "http://placehold.it/350x200"
  )
end
puts "5 skills added"
puts "10 blogs added"
```
- rails db:setup
  - Redoing from scratch
  - Don't do this in the production environment
## 59. Index.html from scratch
- Add `def index` into `app/controllers/portfolios_controller.rb`
```
class PortfoliosController < ApplicationController
  def index
    @portfolio_items = Portfolio.all
  end
end
```
- `app/views/portfolios` folder is produced as empty. Add index.html.erb as a regular html
```
<h1> Porfolio items </h1>
<% @portfolio_items.each do |portfolio_item| %>
<p><%= portfolio_item.title %></p>
<p><%= portfolio_item.subtitle %></p>
<p><%= portfolio_item.body %></p>
<%= image_tag portfolio_item.thumb_image %>
<% end %>
```

## 60. New functionality from scratch
- Add `def new` into `app/controllers/portfolios_controller.rb`
```
class PortfoliosController < ApplicationController
  def index
    @portfolio_items = Portfolio.all
  end
  def new
    @portfolio_item = Portfolio.new
  end
end
```
  - Note that `@portfolio_item` as new.html.erb uses the variable of `@portfolio_item` (not items)
- Add a new file of `app/views/portfolios/new.html.erb`
```
<h1> Create a new Portfolio Item </h1>
<%= form_for(@portfolio_item) do |f| %>
  <div id="field">
    <%= f.label :title %>
    <%= f.text_field :title %>
  </div>
  <div id="field">
    <%= f.label :subtitle %>
    <%= f.text_field :subtitle %>
  </div>
  <div id="field">
    <%= f.label :body %>
    <%= f.text_field :body %>
  </div>
  <div class="actions">
    <%= f.submit %>
  </div>
<% end %>
```
- Now add create function to the controller
```
def create
    @portfolio_item = Portfolio.new(params.require(:portfolio).permit(:title, :subtitle,:body))
    respond_to do |format|
      if @portfolio_item.save
        format.html { redirect_to portfolios_path, notice: 'Your portfolio item is now live' }
        format.json { render :show, status: :created, location: @portfolio_item }
      else
        format.html { render :new }
        format.json { render json: @portfolio_item.errors, status: :unprocessable_entity }
      end
    end
  end
  ```
  - From `http://localhost:3000/portfolios/new`, now you can create new portfolio but it will yield an error message as the thumb image from index.html.erb is missing. So updated index.html.erb with `<%= image_tag portfolio_item.thumb_image unless portfolio_item.thumb_image.nil?%>`, checking if the condition of an image is nil or not

### When following message appears:
```
FATAL: Listen error: unable to monitor directories for changes.
Visit https://github.com/guard/listen/wiki/Increasing-the-amount-of-inotify-watchers for info on how to fix this.
```
- Multiple ruby instances are existing. Terminate them
