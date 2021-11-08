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
- Database -> Collection -> Documents
- Created implicitly

17. The shell & MongdoDB Drivers
- Mongo shell is based on javascript
- Supports C/C#/C++/Python/Ruby/...

18. Creating databases & Collections
```
use flights
db.flightData.insertOne({})
db.flightData    
```
20. JSON vs BSON
- BSON: binary JSON
- MongoDB converts JSON->BSON
    - MongoDB injects extra meta data into BSON
    - `"_id" : ObjectId("XXXXX")`
- Using `_id:"XXX"`, can over-ride the mongoDB given automatic ID. Duplicated ID is not allowed.

22. Finding, inserting, deleting & updating elements
- `db.flightData.deleteOne({departureAirport:"TXL"})`
    - Delete one element only (first element found)
- `db.flightData.updateOne({departureAirport:"TXL"},{$set: {marker: "abc"}})`
    - Find (query) the first element matching `{departureAirport:"TXL"}` then add `marker` key/value
    - `updateMany()` for multiple data

24. Diving deeper into Finding Data
```
db.flightData.find({intercontinental: true}).pretty() # query for true/false
db.flightData.find({distance: {$gt: 1000}}).pretty() # query data whose distance > 1000
```
- To find only one (or the first element), use `findOne()`. `pretty()` doesn't work with `findOne()`

25. update() vs updateMany()
- `db.flightData.update({"_id" : "tx1l-h1-1"},{delayed: true})` will over-write the existing data element. In other words, existing other keys/values will be removed!!!
  - In order to update/replace one key, use `replaceOne()`
- Instead of update(), updateOne() or updateMany() is recommended as it has better mechanism for error message

26. cursor Object
- find() gives you cursor, not document.
- Reducing memory/bandwidth requirement
- TBD

27. Projection
- `db.passengers.find({},{name:1, _id:0}).pretty()`
- Disable `_id` key and others
- Enable `name` key only
- Can reduce overhead like memory usage and bandwidth requirement

29. Embedded Documents
- `db.flightData.updateMany({}, {$set: {status:{description: "on-time", lastUpdated: "1 hour ago"}}})`
- `{}` will be wild card for query
- Injecting nested Documents

30. Arrays in Documents
- `db.passengers.updateOne({name:"Albert Twostone"},{$set: {hobbies: ["sports", "cooking"]}})`
- Use `[]` for array values

Assignment_1
```
# insert 3 patient records
> db.patient.insertMany([
... {
... "firstName": "Max",
... "lastName": "Schwarzmuller",
... "age":29,
... "history": [
... {"disease": "cold", "treatment": "shot"},
... {"surgery required": false}]
... },
... {
... "firstName": "Miku",
... "lastName": "Hatsune",
... "age":17,
... "history": [
... {"disease": "cold", "treatment": "vitamin"},
... {"surgery required": true}]
... },
...
... {
... "firstName": "Teto",
... "lastName": "Masaru",
... "age":31,
... "history": [
... {"disease": "constipation", "treatment": "water"},
... {"surgery required": false}]
... }
... ]
... )
# Update patient data of 1 patient with new age, name and history
> db.patient.update({firstName:"Teto"}, {age:13,firstName:"Momo","lastName":"Minky", history:[{"disease":"flu","vaccinated":false},{"surgery done":false}]})
WriteResult({ "nMatched" : 1, "nUpserted" : 0, "nModified" : 1 })
# Find all patients who are older than 24
> db.patient.find({age: {$gt: 24}}).pretty()
# Delete all patients with cold as disease
> db.patient.deleteMany({"history.disease":"cold"})
```
