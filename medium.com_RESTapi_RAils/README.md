# Creating a REST API with Rails
- By Oliver DS
- Ref: https://medium.com/@oliver.seq/creating-a-rest-api-with-rails-2a07f548e5dc

## Getting started
- rails new rest-api-guide --api --database=postgresql

## Models and Controllers
- cd res-api-guide
- rails generate model User username:string password:string
- rails generate model Fact user:references fact:string likes:integer
- apps/models/user.rb
```
class User < ApplicationRecord
  has_many :facts
end
```
- apps/models/fact.rb
```
class fact < ApplicationRecord
  belongs_to :user
end
```
- Edit config/database.yml
```
host: localhost
username: postgres
password: XXXXX
```
- rails db:setup
- rails db:migrate
  - schema.rb is produced at db/ folder

## Controllers
- rails g controller api/v1/Users
- rails g controller api/v1/Facts
  - v1 for version1
  - Note that *User* for model while *Users* for controllers. This is a syntactical requirement of Rails
- Check config/routes.rb
```
Rails.application.routes.draw do
  namespace :api do
    namespace :v1 do
      resources :users do
        resources :facts
      end
    end
  end
end
```
- rails routes
- Users controller with RESTful architecture
  - Update app/controllers/api/v1/users_controller.rb
```
class Api::V1::UsersController < ApplicatinoController
  # get user
  def index
    @users = User.all
    render json: @users
  end
  # get /user/:id
  def show
    @user  = User.find(params[:id])
    render json: @user
  end
  # post /users
  def create
    @user = User.new(user_params)
    if @user.save
      render json: @user
    else
      render error: { error: 'Unable to create user.' }, status: 400
    end
  end
  # put /users:
  def update
    @user = User.find(params[:id])
    if @user
      @user.update(user_params)
      render json: { message: 'User successfully updated.' }, status: 200
    else
      render json: { error: 'Unable to update User.' }, status: 400
    end
  end
  # delete /users/:id
  def destroy
    @user = User.find(params[:id])
    if @user
      @user.destroy
      render json: { message: 'User successfully deleted.' }, status: 200
    else
      render json: { error: 'Unable to delete User.' }, status: 400
    end
  end
  private
  def user_params
    params.require(:user).permit(:username, :password)
  end
end
```
- `user_params` is private for security purposes
- Facts controller with RESTful architecture
  - Update app/controllers/api/v1/facts_controller.rb
```
class Api::V1::FactsController < ApplicatinoController
  before_action :find_fact, only: [:show, :update, :destory]
  def index
    @facts = Fact.all
    render json: @facts
  end
  def show
    render json: @fact
  end
  def create
    @fact = Fact.new(fact_params)
    if @fact.save
      render json: @fact
    else
      render error: { error: 'Unable to create fact.' }, status: 400
    end
  end
  def update
    if @fact
      @fact.update(fact_params)
      render json: { message: 'Fact successfully updated.' }, status: 200
    else
      render json: { error: 'Unable to update fact.' }, status: 400
    end
  end
  def destroy
    if @fact
      @fact.destroy
      render json: { message: 'Fact successfully deleted.'}, status: 200
    else
      render json: { error: 'Unable to delete fact.' }, status: 400
    end
  end
  private
  def fact_params
    params.require(:fact).permit(:fact, :likes, :user_id)
  end
  def find_fact
    @fact = Fact.find(params[:id])
  end
end
```
- rails s # or rails server
- Open a web-browser and open http://localhost:3000/api/v1/users
  - An empty json [] will appear
- rails c # or rails console
```
irb(main):001:0> oliver = User.create(username: 'Oliver', password: 'Oliverpass')
...
irb(main):002:0> fact_one = Fact.create(fact: 'Wiley Hardeman post was the first pilot to fly solo around the world.', likes:1, user_id:1)
...
irb(main):003:0> fact_two = Fact.create(fact: 'The Symphony No1 in E flat major, k.16 was written by Mozart at the age of 8.', likes:2, user_id:1)
```
- Updated table will be shown at http://localhost:3000/api/v1/users and http://localhost:3000/api/v1/users/1/facts
