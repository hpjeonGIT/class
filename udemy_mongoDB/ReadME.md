6. General Setup Instructions
- We are going to use: Community version, Compass for GUI.
- `sudo apt install ./mongodb-org-server_5.0.3_amd64.deb`
  - command: `mongod --dbpath ~/hw/class/udemy_mongoDB/mongodb/data --logpath  ~/hw/class/udemy_mongoDB/mongodb/logs/mongo.log`
  - The folder must exist
- `sudo apt install mongodb-org-shell_5.0.3_amd64.deb`
```
$ mongo
MongoDB shell version v5.0.3
...
> show dbs
admin   0.000GB
config  0.000GB
local   0.000GB
> exit
bye
```
- `tar xvf mongosh-1.1.1-linux-x64.tgz`
  - move the folder to the install directory
  - This is shell for javascript (?)
- `tar zxf mongodb-database-tools-debian10-x86_64-100.5.1.tgz`
  - Tools using mongoimport

10. Time to Get Started
```
> showdbs
uncaught exception: ReferenceError: showdbs is not defined :
@(shell):1:1
> show dbs
admin   0.000GB
config  0.000GB
local   0.000GB
> use shop
switched to db shop
> db.products.insertOne({name:"Max",price:12.99})
{
	"acknowledged" : true,
	"insertedId" : ObjectId("61854133c2a541c9767d806b")
}
> db.products.find()
{ "_id" : ObjectId("61854133c2a541c9767d806b"), "name" : "Max", "price" : 12.99 }
> db.products.insertOne({name:"A T-shirt",price:29.99, "despcription":"foo one"})
{
	"acknowledged" : true,
	"insertedId" : ObjectId("61854199c2a541c9767d806c")
}
> db.products.find().pretty()
{
	"_id" : ObjectId("61854133c2a541c9767d806b"),
	"name" : "Max",
	"price" : 12.99
}
{
	"_id" : ObjectId("61854199c2a541c9767d806c"),
	"name" : "A T-shirt",
	"price" : 29.99,
	"despcription" : "foo one"
}
>
```

15. Module introduction
- CRUD: Create, Read, Update, Delete

16. Understanding Databases, Collections and Documents
- Datebase -> Collection -> Documents
- Created implicitly

