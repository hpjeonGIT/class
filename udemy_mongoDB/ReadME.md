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
## history.disease needs double-quotation marks.
```

34. Resetting database
- list of current database: `show dbs`
- delete the database: `use test`, `db.dropDatabase()`
    - Semicolon (;) doesn't split commands in one line
    - or `db.test.drop()`
- list of collections in the current database: `db.getCollectionNames()`

36. Schemas
- MongoDB doesn't enforce Schemas but still can use Schemas.

37. Structuring Documents
- For empty value, use `null`

38. Data types
- Text
- Boolean: `true`, `false`
- Number: Integer for int32, NumberLong for Int64, NumberDecimal for high precision. Regular number is stored as float
- ObjectId:
- ISODate/Timestamp
- Embedded Document: not external document but nested `document` data
- Array

39. Data types in action
- `new Date()` as value produces `ISODate("2021-11-09T15:22:53.077Z")`
- `new Timestamp()` as value produces `Timestamp(1636471373, 1)`
- 12345678901234567890 will be truncated as it is larger than regular number(float) in javascript
```
> db.stats()
{
	"db" : "companyData",
	"collections" : 1,
	"views" : 0,
	"objects" : 1,
	"avgObjSize" : 233,
	"dataSize" : 233,
	"storageSize" : 20480,
	"freeStorageSize" : 0,
	"indexes" : 1,
	"indexSize" : 20480,
	"indexFreeStorageSize" : 0,
	"totalSize" : 40960,
	"totalFreeStorageSize" : 0,
	"scaleFactor" : 1,
	"fsUsedSize" : 469681377280,
	"fsTotalSize" : 715936817152,
	"ok" : 1
}
```
- Using regular number `123` or `NumberInt(123)` will change the dataSize from `db.stats()`
- Embedding or nested documents are allowed up to 100
- int32 up to 2,147,483,647
- int64 up to 9,223,372,036,854,775,807
- Text up to 16MB

49. Relations
- Embedded document vs. references (using ObjectId())

50. Joining with `$lookup`
```
> use bookRegistry
switched to db bookRegistry
## Embedding document
>  db.books.insertOne({name:"My book", authors:[{name: "Max R", age:31},{name:"Manuel X", age:51}]})
{
	"acknowledged" : true,
	"insertedId" : ObjectId("618bcceedfe0ac1c05978bcb")
}
> db.books.find().pretty()
{
	"_id" : ObjectId("618bcceedfe0ac1c05978bcb"),
	"name" : "My book",
	"authors" : [
		{
			"name" : "Max R",
			"age" : 31
		},
		{
			"name" : "Manuel X",
			"age" : 51
		}
	]
}
> db.authors.insertMany([{name:"Max R", age:31, address: {street: "Main"}}, {name: "Manuel X", age:51, address: {street: "Elm"}}])
{
	"acknowledged" : true,
	"insertedIds" : [
		ObjectId("618bcd82dfe0ac1c05978bcc"),
		ObjectId("618bcd82dfe0ac1c05978bcd")
	]
}
## Updating with references
> db.books.updateOne({}, {$set: {authors: [ObjectId("618bcd82dfe0ac1c05978bcc"),ObjectId("618bcd82dfe0ac1c05978bcd")]}})
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }
> db.books.find().pretty()
{
	"_id" : ObjectId("618bcceedfe0ac1c05978bcb"),
	"name" : "My book",
	"authors" : [
		ObjectId("618bcd82dfe0ac1c05978bcc"),
		ObjectId("618bcd82dfe0ac1c05978bcd")
	]
}
# from: COLLECTION, localField: KEYWORD
# Doesn't store to books collection
> db.books.aggregate([{$lookup: {from: "authors", localField: "authors", foreignField: "_id", as:"creators"}}]).pretty()
{
	"_id" : ObjectId("618bcceedfe0ac1c05978bcb"),
	"name" : "My book",
	"authors" : [
		ObjectId("618bcd82dfe0ac1c05978bcc"),
		ObjectId("618bcd82dfe0ac1c05978bcd")
	],
	"creators" : [
		{
			"_id" : ObjectId("618bcd82dfe0ac1c05978bcc"),
			"name" : "Max R",
			"age" : 31,
			"address" : {
				"street" : "Main"
			}
		},
		{
			"_id" : ObjectId("618bcd82dfe0ac1c05978bcd"),
			"name" : "Manuel X",
			"age" : 51,
			"address" : {
				"street" : "Elm"
			}
		}
	]
}
>
```

52. Implementing Blog
```
> use blog
switched to db blog
> db.users.insertMany([{name:"Max S", age:29, email: "myemail@some.come"}, {name:"Manuel R", age:51, email :"manu@test.com"}])
{
	"acknowledged" : true,
	"insertedIds" : [
		ObjectId("618bd17edfe0ac1c05978bce"),
		ObjectId("618bd17edfe0ac1c05978bcf")
	]
}
> db.users.find().pretty()
{
	"_id" : ObjectId("618bd17edfe0ac1c05978bce"),
	"name" : "Max S",
	"age" : 29,
	"email" : "myemail@some.come"
}
{
	"_id" : ObjectId("618bd17edfe0ac1c05978bcf"),
	"name" : "Manuel R",
	"age" : 51,
	"email" : "manu@test.com"
}
> db.posts.insertOne({title: "My first Post!", text: "Some contents here", tags: ["new one", "text", "test01"], creators: ObjectId("618bd17edfe0ac1c05978bcf"), comments: [{text: " I like this post!", author: ObjectId("618bd17edfe0ac1c05978bce")}]})
{
	"acknowledged" : true,
	"insertedId" : ObjectId("618bd213dfe0ac1c05978bd0")
}
> db.posts.findOne()
{
	"_id" : ObjectId("618bd213dfe0ac1c05978bd0"),
	"title" : "My first Post!",
	"text" : "Some contents here",
	"tags" : [
		"new one",
		"text",
		"test01"
	],
	"creators" : ObjectId("618bd17edfe0ac1c05978bcf"),
	"comments" : [
		{
			"text" : " I like this post!",
			"author" : ObjectId("618bd17edfe0ac1c05978bce")
		}
	]
}
```

53. Collection document validation
```
db.createCollection('posts', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['title', 'text', 'creator', 'comments'],
      properties: {
        title: {
          bsonType: 'string',
          description: 'must be a string and is required'
        },
        text: {
          bsonType: 'string',
          description: 'must be a string and is required'
        },
        creator: {
          bsonType: 'objectId',
          description: 'must be an objectid and is required'
        },
        comments: {
          bsonType: 'array',
          description: 'must be an array and is required',
          items: {
            bsonType: 'object',
            required: ['text', 'author'],
            properties: {
              text: {
                bsonType: 'string',
                description: 'must be a string and is required'
              },
              author: {
                bsonType: 'objectId',
                description: 'must be an objectid and is required'
              }
            }
          }
        }
      }
    }
  }
});
> db.posts.insertOne({title: "My first Post!", text: "Some contents here", tags: ["new one", "text", "test01"], creators: ObjectId("618bd17edfe0ac1c05978bcf"), creator: ObjectId("618bd17edfe0ac1c05978bcf"), comments: [{text: " I like this post!", author: ObjectId("618bd17edfe0ac1c05978bce")}]})
{
	"acknowledged" : true,
	"insertedId" : ObjectId("618bd647dfe0ac1c05978bd4")
}
```
- When the new data doesn't match the schema, there will be an error of `"errmsg" : "Document failed validation",`
- Missing key names are listed in "missingProperties"
- Or run validator:
```
> db.runCommand({collMod: "posts", validator: {
...     $jsonSchema: {
...       bsonType: 'object',
...       required: ['title', 'text', 'creator', 'comments'],
...       properties: {
...         title: {
...           bsonType: 'string',
...           description: 'must be a string and is required'
...         },
...         text: {
...           bsonType: 'string',
...           description: 'must be a string and is required'
...         },
...         creator: {
...           bsonType: 'objectId',
...           description: 'must be an objectid and is required'
...         },
...         comments: {
...           bsonType: 'array',
...           description: 'must be an array and is required',
...           items: {
...             bsonType: 'object',
...             required: ['text', 'author'],
...             properties: {
...               text: {
...                 bsonType: 'string',
...                 description: 'must be a string and is required'
...               },
...               author: {
...                 bsonType: 'objectId',
...                 description: 'must be an objectid and is required'
...               }
...             }
...           }
...         }
...       }
...     }
...   }
... })
{ "ok" : 1 }
```

71. Insert()
- Instead of insert(), use insertOne() or insertMany()

72. Ordered insert()
- `> db.hobbies.insertMany([{_id: "sports", name: "Sports"}, {_id: "cooking", name: "Cooking"}, {_id: "cars", name: "Cars"}])`: works OK
- `> db.hobbies.insertMany([{_id: "yoga", name: "Yoga"}, {_id: "cooking", name: "Cooking"}, {_id: "cars", name: "Cars"}])`: breaks as duplicated `_id` found. But `yoga` is still added to collections as the failure happens at 2nd document, canceling the left-over
- `db.hobbies.insertMany([{_id: "yoga", name: "Yoga"}, {_id: "cooking", name: "Cooking"}, {_id: "hiking", name: "Hiking"}], {ordered: false})`
    - By using `{ordered: false}`, failed documents are ignored but working elements are added into collections
    - Default status of `ordered` is true
- Even though insertMany() may fail, accepted documents are not rolled-back regardless of ordered or unordered

73. writeConcern
- `{w: 1, j: undefined, wtimeout:200}`: w for write and j for journal. May set timeout
- Ex) `> db.persons.insertOne({name: "Chris", age:41}, {writeConcern: {w:1, j: true, wtimeout:100}})`

75. Assignment2
```
use companyData
db.companies.insertOne({_id: "glass", name: "Glass inc", budget: 100})
db.companies.insertMany([{_id: "glass", name: "fiber inc", budget: 333}, {_id: "ceram", name: "Ceramic Inc", budget: 200} ], {ordered:false})
db.companies.insertOne({name: "brick company", budget:4444}, {writeConcern: {w:1 , j : true }} )
db.companies.insertOne({name: "shoes company", budget:55}, {writeConcern: {w:1 , j : false }} )
```

76. Importing data
- `mongoimport mydata.json -d dbase_name -c collexn_name --jsonArray --drop`
    - `--drop` will drop the database if it exists already

80. Methods, filters, and operators
- Regarding `db.myCollection.find( { age: 32} )`
    - current database: db
    - access this collection: mycollection
    - method: find()
    - filter: `{ age: 32}`
        - field(key) : age
        - value: 32
- Regarding `db.myCollection.find( { age:  {$gt : 30 }} )`
    - current database: db
    - access this collection: mycollection
    - method: find()
    - range filter: `{ age:  {$gt : 30 }}`
        - operator: `$gt`

85. Querying embedded fields and arrays
```
	"rating" : {
		"average" : 8.1
	},
```
- use double quotation: `db.shows.find({"rating.average": {$gt: 8}})`
```
	"genres" : [
		"Comedy",
		"Family"
	],
```
- `db.shows.find({genres: [ "Comedy" ]}).pretty()`

86. `$in` vs `$nin`
- `db.shows.find({ runtime: {$in: [42,30] }}).pretty()`
	- Find runtime of 42 or 30. Not range of 30-42
- `$nin` is `not $in`

87. `$or` vs `$nor`
```
> db.shows.find({$or: [{"rating.average": {$lt:5} }, {"rating.average": {$gt: 9.3} } ]}).count()
4
```

88. `$and`
```
> db.shows.find({$and: [{"rating.average": {$gt:9} }, {genres: "Drama"}] }).count()
3
> db.shows.find({"rating.average": {$gt:9} , genres: "Drama"}).count()
3
```
- Two queries in a single `{}`
- `db.shows.find({"rating.average": {$gt:9}} , {genres: "Drama"})` gives a wrong result (only the first query)
	- However, if same field/key is used, second field will overwrite the first query
```
> db.shows.find({genres: "Drama", genres: "Horror"}).count()
23
> db.shows.find({genres: "Horror"}).count()
23
```	
- To find two matching values of the same field, we need `$and`
```
> db.shows.find({$and: [{genres: "Drama"}, {genres: "Horror"}]}).count()
17
```

90. Element operator
```
> db.users.insertMany([ {name: "Max", hobbies:[ {title: "sports", frequency:3 }, {title: "cooking", frequency: 6}], phone: 0132456789}, {name: "Manuel", hobbies: [ {title: "Cars"}, {freqeency: 7}], phone : "945672381", age: 30 } ])
> db.users.find({age: {$exists: true}}).count()
1
> db.users.insertOne( {name: "Anna", hobbies: {title: "yoga", frequency:3 }, phone: "753159", age:null})
> db.users.find({age: {$exists: true}}).count()
2
## <- null is still recognized as existing
> db.users.find({age: {$exists: true, $ne: null}}).count()
1
## <- user one more query to check null
```

91. `$type`
```
> db.users.find({phone: {$type: "string"}}).count()
3
> db.users.find({phone: {$type: "double"}}).count()
2
> db.users.find({phone: {$type: ["string", "double"]}}).count()
5
```

92. `$regex`
```
> db.shows.find({summary: {$regex: /musical/}}).count()
2
```

93. `$expr`
```
> db.sales.insertMany([{volume:100, target:120}, {volume:89,targe:80}, {volume:200, target:177}])
> db.sales.find({$expr: {$gt: ["$volume", "$target"]}})
## find documents where volume > target
{ "_id" : ObjectId("618eb30ef02d1be5657376af"), "volume" : 89, "targe" : 80 }
{ "_id" : ObjectId("618eb30ef02d1be5657376b0"), "volume" : 200, "target" : 177 }
> db.sales.find({$expr: {$gt: [ {$cond:  {if:  {$gte: ["$volume", 190] }, then: {$subtract: ["$volume", 10]}, else: "$volume"  } }, "$target"  ]}}).pretty()
## find documents where 1) when volume >190 (volume -10) or 2) volume must be larger than target
{
	"_id" : ObjectId("618eb30ef02d1be5657376af"),
	"volume" : 89,
	"targe" : 80
}
{
	"_id" : ObjectId("618eb30ef02d1be5657376b0"),
	"volume" : 200,
	"target" : 177
}

```
- Q: what if target field doesn't exist?

Assignment 3.
- Import boxoffice.json
- Search all movies that have a rating higher than 9.2 and a runtime lower than 100 min
- Search all movies that have a genre of "drama" or "action"
- Search all movies where visitors exceeded expectedVisitors
```
> db.boxoff.find({$and: [{"meta.rating": {$gt: 9.2}}, {"meta.runtime": {$lt: 100}}]}).count()
1
> db.boxoff.find({$or: [{genre: "drama"}, {genre: "action"}]}).count()
3
> db.boxoff.find({$expr: {$gt: ["visitors", "expectedVisitors"]}}).count()
3
```
