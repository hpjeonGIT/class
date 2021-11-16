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
## -> this is wrong
> db.boxoff.find({$expr: {$gt: ["$visitors", "$expectedVisitors"]}}).count()
1
```
- For expression, **add `$` to the  field/key name**

95. Querying array size with `$size`
```
> db.users.find({hobbies: {$size: 2}})
```
- Exact match only. Larger or less is not supported yet

96. Querying arrays with `$all`
- `> db.boxoff.find({genre: ["action", "thriller"]})` will find `"action", "thriller"`, not `"thriller", "action"`
	- Exact match only
- Using `all` will search document containing items regardless of the order or more elements
	- `> db.boxoff.find({genre: {$all: ["action", "thriller"]}})`

97. Querying arrays with `$elemMatch`
```
> db.users.insertMany([ {name:"Anna", hobbies: [ {title: "Sports", frequency:3}, {title :"Yoga", frequency:6} ]}, {name:"David", hobbies: [{title: "Sports", frequency:2}, {title:"Yoga", frequency: 5}] }])
> db.users.find({$and: [{"hobbies.title": "Sports"}, {"hobbies.frequency": {$gte: 5}}]})
{ "_id" : ObjectId("61925c27f02d1be5657376b8"), "name" : "David", "hobbies" : [ { "title" : "Sports", "frequency" : 2 }, { "title" : "Yoga", "frequency" : 5 } ] }
## -> This will find David as well using frequency of Yoga, not Sports
> db.users.find({hobbies: {$elemMatch: {title: "Yoga", frequency:{$gte: 6}}}})
{ "_id" : ObjectId("61925ccbf02d1be5657376b9"), "name" : "Anna", "hobbies" : [ { "title" : "Sports", "frequency" : 3 }, { "title" : "Yoga", "frequency" : 6 } ] 
## -> Using $elemMatch, field/key name is limited within the nested document
```

Assignment 4
- Import attached data into boxOffice database
- Find all movies with exactly two genres
- Find all movies which aired in 2018
- Find all movies having ratings greater than 8 but lower than 10
```
> db.movie.find({genre: {$size:2}})
> db.movie.find({"meta.aired": {$eq: 2018}})
> db.movie.find({$and: [{"ratings": {$gt:8}}, {"ratings": {$lt:10}}]})
## from the solution:
> db.movie.find({ratings: {$elemMatch: {$gt: 8, $lt:10}}})
```

99. Cursors
```
> const cursor = db.shows.find()
> cursor.hasNext()
true
```
100. Sorting
```
> db.shows.find().sort({"rating.average":1, runtime:-1}).pretty()
```

101. Skipping and limiting
- .skip(10)
- .limit(10)
```
> db.dataSimple.find().skip(100).limit(5)
{ "_id" : ObjectId("61926a0ff02d1be565737722"), "id" : 100, "rating" : 5 }
{ "_id" : ObjectId("61926a0ff02d1be565737723"), "id" : 101, "rating" : 3 }
{ "_id" : ObjectId("61926a0ff02d1be565737724"), "id" : 102, "rating" : 4 }
{ "_id" : ObjectId("61926a0ff02d1be565737725"), "id" : 103, "rating" : 6 }
{ "_id" : ObjectId("61926a0ff02d1be565737726"), "id" : 104, "rating" : 10 }
> 
```

102. Projection
- 1 to show while 0 to hide
```
> db.shows.find({} , {name:1, type:1, _id:0})
{ "name" : "Under the Dome", "type" : "Scripted" }
{ "name" : "Person of Interest", "type" : "Scripted" }
```

103. Projection in array

104. `$slice`
```
> db.shows.find({}, {genres: 1,  name:1}).limit(2).pretty()
{
	"_id" : ObjectId("618e7ee2509a74d40d7771e7"),
	"name" : "Under the Dome",
	"genres" : [
		"Drama",
		"Science-Fiction",
		"Thriller"
	]
}
{
	"_id" : ObjectId("618e7ee2509a74d40d7771e8"),
	"name" : "Person of Interest",
	"genres" : [
		"Drama",
		"Action",
		"Crime"
	]
}
> db.shows.find({}, {genres: {$slice:[1,2]}, name:1}).limit(2).pretty()
{
	"_id" : ObjectId("618e7ee2509a74d40d7771e7"),
	"name" : "Under the Dome",
	"genres" : [
		"Science-Fiction",
		"Thriller"
	]
}
{
	"_id" : ObjectId("618e7ee2509a74d40d7771e8"),
	"name" : "Person of Interest",
	"genres" : [
		"Action",
		"Crime"
	]
}
```
- Showing 1-2 elements only, hiding 0th element

107. `updateOne()`, `updateMany()`, and `$set`
```
> db.users.insertOne({name:"Chris", hobbies: ["Sports", "Cooking", "Hiking"]})
> db.users.updateOne({_id:ObjectId("61926f1bf02d1be565737aa6")}, {$set: {hobbies: [{title:"Sports", frequency:5}, {title: "Cooking", frequency:3}, {title:"Hiking", frequency:1}]}})
```
- Using `$set` adds elements (not removing existing fields)
```
> db.users.updateMany({"hobbies.title":"Sports"}, {$set: {isSporty: true}})
```
- Adds the field of `isSporty`

108. Multiple-fields with `$set`
```
> db.users.updateMany({"hobbies.title":"Sports"}, {$set: {isSporty: true, isCooky: false}})
```

109. Increment or decrement values
```
> db.users.updateMany({_id:ObjectId("618e88d4f02d1be5657376ac")}, {$inc: {age:100}})
> db.users.updateOne({_id:ObjectId("618e88d4f02d1be5657376ac")}, {$inc: {age:100}, $set: {hasHouse: false } })
```
- There is no `$dec` for decrement. Use negative number with `$inc`

110. `$min`, `$max`, `$mul`
```
> db.users.find({_id:ObjectId("618e88d4f02d1be5657376ac")})
{ "_id" : ObjectId("618e88d4f02d1be5657376ac"), "name" : "Manuel", "hobbies" : [ { "title" : "Cars" }, { "freqeency" : 7 } ], "age" : 130, "phone" : "945672381", "hasHouse" : false }
> db.users.updateOne({_id:ObjectId("618e88d4f02d1be5657376ac")}, {$min: {age:38}})
## -> update age as min(current value, 38). Updated from 130->38
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }
> db.users.find({_id:ObjectId("618e88d4f02d1be5657376ac")})
{ "_id" : ObjectId("618e88d4f02d1be5657376ac"), "name" : "Manuel", "hobbies" : [ { "title" : "Cars" }, { "freqeency" : 7 } ], "age" : 38, "phone" : "945672381", "hasHouse" : false }
> db.users.updateOne({_id:ObjectId("618e88d4f02d1be5657376ac")}, {$min: {age:40}})
## -> update age as min(current value, 40). 38 is maintained.
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 0 }
> db.users.find({_id:ObjectId("618e88d4f02d1be5657376ac")})
{ "_id" : ObjectId("618e88d4f02d1be5657376ac"), "name" : "Manuel", "hobbies" : [ { "title" : "Cars" }, { "freqeency" : 7 } ], "age" : 38, "phone" : "945672381", "hasHouse" : false }
> db.users.updateOne({_id:ObjectId("618e88d4f02d1be5657376ac")}, {$max: {age:48}})
## -> age as max(current value, 48)
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }
> db.users.updateOne({_id:ObjectId("618e88d4f02d1be5657376ac")}, {$mul: {age:1.1}})
## -> age = age*1.1
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }
> db.users.find({_id:ObjectId("618e88d4f02d1be5657376ac")}).pretty()
{
	"_id" : ObjectId("618e88d4f02d1be5657376ac"),
	"name" : "Manuel",
	"hobbies" : [
		{
			"title" : "Cars"
		},
		{
			"freqeency" : 7
		}
	],
	"age" : 52.800000000000004,
	"phone" : "945672381",
	"hasHouse" : false
}
```

111. Removing fields
- Use `$unset` to remove field 
```
> db.companies.find()
{ "_id" : "glass", "name" : "Glass inc", "budget" : 100 }
{ "_id" : "ceram", "name" : "Ceramic Inc", "budget" : 200 }
{ "_id" : ObjectId("618d8b49cef57a0b0c0e3efc"), "name" : "brick company", "budget" : 4444 }
{ "_id" : ObjectId("618d8b4bcef57a0b0c0e3efd"), "name" : "shoes company", "budget" : 55 }
> db.companies.updateMany({}, {$unset: {budget:""}})
{ "acknowledged" : true, "matchedCount" : 4, "modifiedCount" : 4 }
> db.companies.find()
{ "_id" : "glass", "name" : "Glass inc" }
{ "_id" : "ceram", "name" : "Ceramic Inc" }
{ "_id" : ObjectId("618d8b49cef57a0b0c0e3efc"), "name" : "brick company" }
{ "_id" : ObjectId("618d8b4bcef57a0b0c0e3efd"), "name" : "shoes company" }
```

112. Renaming fields
```
> db.companies.updateMany({}, {$rename: {name: "Name"}})
```

113. `upsert` in updateOne/Many()
```
> db.users.updateOne({name: "Maria"}, {$set: {age:29, hobbies: [{title: "organic", frequency:3}, {title: "yoga", frequency:5}]}})
{ "acknowledged" : true, "matchedCount" : 0, "modifiedCount" : 0 }
## updateOne() is not effective as "Maria" doesn't exist
> db.users.updateOne({name: "Maria"}, {$set: {age:29, hobbies: [{title: "organic", frequency:3}, {title: "yoga", frequency:5}]}}, {upsert: true})
{
	"acknowledged" : true,
	"matchedCount" : 0,
	"modifiedCount" : 0,
	"upsertedId" : ObjectId("6193be6af72795f8ecd822d2")
}
## using upsert true, new data is added if request one doesn't exist
```

Assignment 5
- Create a new collection sports and upsert two new documents with fields of title, requiresTeam
- Update all document which do require a team by adding a new field with the minimum amount of players
- Update all documents that require a team by increasing the number of required players by 10
```
> db.sports.insertMany([{title:"baseball", requiresTeam: true}, {title:"tennis", requiresTeam: true}]
> db.sports.insertOne({title:"workout", requiresTeam: false})
> db.sports.updateOne({title:"soccer"}, {$set: {requiresTeam: true}}, {upsert:true}) 
> db.sports.updateMany({"requiresTeam":true}, {$set: {minimumPlayers:10}})
{ "acknowledged" : true, "matchedCount" : 2, "modifiedCount" : 2 }
> db.sports.updateMany({"requiresTeam":true}, {$inc: {minimumPlayers:10}})
{ "acknowledged" : true, "matchedCount" : 2, "modifiedCount" : 2 }
```
- Solution from the lecture
```
db.sports.updateMany({}, {$set: {title: "Football", requiredTeam: true}}, {upsert: true})
## -> This will corrupt existing sports collection. Drop it beforehand
db.sports.updateMany({title: "Soccer"}, {$set: {requiredTeam: true}}, {upsert: true})

```

114. Updating array elements
- Use `$elemMatch` to find exact documents
- Use `.$.` as a place-holder to locate the matched element from the query
	- This place-holder works for only the **first element only**. For all elements, use `.$[].`
	- All of `hobbies` must be array. If non-array `hobbies` exist, this will not work
```
> db.users.find({hobbies: {$elemMatch: {title: "Sports", frequency: {$gte:5}}}})
> db.users.updateMany({hobbies: {$elemMatch: {title: "Sports", frequency: {$gte:5}}}}, {$set: {"hobbies.$.highFrequency":true }}) 
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }
> db.users.find({hobbies: {$elemMatch: {title: "Sports", frequency: {$gte:5}}}}).pretty()
{
	"_id" : ObjectId("61926f1bf02d1be565737aa6"),
	"name" : "Chris",
	"hobbies" : [
		{
			"title" : "Sports",
			"frequency" : 5,
			"highFrequency" : true
		},
		{
			"title" : "Cooking",
			"frequency" : 3
		},
		{
			"title" : "Hiking",
			"frequency" : 1
		}
	],
	"isSporty" : true
}
```

115. Updating all array elements
```
> db.users.updateMany({"hobbies.frequency": {$gt:2}}, {$set: {"hobbies.$[].goodFrequency": true}})
```

116. arrayFilters
```
> db.users.updateMany({"hobbies.frequency": {$gt:2}}, {$set: {"hobbies.$[el].betterFrequency": true}}, {arrayFilters: [{"el.frequency": {$gt:3 }}] })
```

117. Adding element to array
- Use `$push`
```
> db.users.updateOne({name:"Maria"}, {$push: {hobbies: {title: "Sports", frequency:2 }}})
```

118. Removing elements from array
- Use `$pull` or `$pop`
```
> db.users.updateOne({name:"Maria"}, {$pull: {hobbies: {title: "organic"} }})
## -> removes matching element in the array
> db.users.updateOne({name:"Maria"}, {$pop: {hobbies:1}})
## -> removes the last element in the array
> db.users.updateOne({name:"Maria"}, {$pop: {hobbies:-1}})
## -> removes the first element in the array
```

123. `deleteOne()` vs `deleteMany()`

124. indexes
- Field/values are extracted and ordered to provide indexes
- Watchout the overhead
