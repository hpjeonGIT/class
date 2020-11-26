## SQL and PostgreSQL: The complete Developer's Guide
- By Stephen Grider

## 3. Creating Tables
- Use pg-sql.com to get a free posgresql database + run queries
```
CREATE TABLE cities (
  name VARCHAR(50),
  country VARCHAR(50),
  population INTEGER,
  area INTEGER
);
```

## 5. Inserting Data into a Table
```
INSERT INTO cities (name, country, population, area)
VALUES ('Tokyo', 'Japan', 38505000, 8223);
INSERT INTO cities (name, country, population, area)
VALUES ('Delhi', 'India', 28125000, 2240),
('Shanghai',  'China',    22125000, 4015),
('Sao Paulo', 'Brazil',   20935000, 3043);
```

## 6. Retrieving data with select
```
SELECT * FROM cities;
SELECT name, area, name, population FROM cities;
```

## 7. Calculated columns
- Math operators
  - +/-/*//
  - ^ : exponent
  - |/ : square root
  - @ : absolute value
  - % : remainder
```
SELECT name, population/area AS population_density FROM cities;
```

## 10. String operators
- || : joins two strings
- CONCAT() : joins multiple strings
- LOWER() : yields a lower case string
- UPPER() : yields a upper case string
- LENGTH() : yields the number of characters in a string
```
SELECT name || ', ' || country FROM cities;
SELECT
  CONCAT(UPPER(name), ', ' , UPPER(country)) AS location
FROM cities;
```

## 11. Filtering with WHERE
- Comparison operator
  - <> : not equal
  - != : not equal
  - = : equal (not ==)
  - >, <, >=, <=, BETWEEN, NOT IN, IN
```
SELECT name, area FROM cities WHERE area > 4000;
SELECT name, area FROM cities WHERE area BETWEEN 2000 AND 4000;
SELECT name, area FROM cities WHERE name IN ('Delhi', 'Shanghai');
SELECT name, area FROM cities WHERE name NOT IN ('Delhi', 'Shanghai');
SELECT name, area FROM cities WHERE area NOT IN (3043,8223) OR name = 'Delhi';
```

## 18. WHERE with calculations
```
SELECT
  name,
  population / area AS population_density
FROM
  cities
WHERE
  population / area > 6000;
```

## 20. Updating rows
```
UPDATE cities
SET population = 39505000
WHERE name = 'Tokyo';
```

## 21. Deleting rows
```
DELETE FROM cities WHERE name='Tokyo';
```

## 25. A sample database design
- One to many relationship: a user -> photos
- Many to one relationship: photos -> a user

## 28. Primary/Foreign keys
- Primary key : unique identifier
- Foreign key : identifier for other links
- Many side has foreign keys
  - One-to-many or many-to-one or many-to-many

## 30. Auto-generated IDs
```
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(50)
  );
INSERT INTO users (username)
VALUES ('john'), ('Mia'), ('Loui'), ('Coco');
```
- Keyword SERIAL PRIMARY

## 31. Generating Foreign keys
```
CREATE TABLE photos(
  id SERIAL PRIMARY KEY,
  url VARCHAR(200),
  user_id INTEGER REFERENCES users(id)
);
INSERT INTO photos(url, user_id)
VALUES
('http://one.jpg', 4);
```
- Keyword REFERENCES
  - Couples with id of users table
  - Works as a constraint

## 32. Taste of JOIN
```
select url, username FROM photos
JOIN users ON users.id = photos.user_id;
```

## 35. Data constraints
- What if user_id in the photos table doesn't match with the id of users table?
```
INSERT INTO photos(url, user_id)
VALUES ('http://jpg',9999);
```
- Produces:
```
insert or update on table "photos" violates foreign key constraint "photos_user_id_fkey"
```
- NULL works OK
```
INSERT INTO photos(url, user_id)
VALUES ('http://jpg',NULL);
```

## 36-40. Constraints around deletion
- Deleting id 1 from users -> produces dangling user_id in photos
- On delete options
  - ON DELETE RESTRICT (default) -> throws an error
  - ON DELETE NO ACTION -> throws an error
  - ON DELETE CASCADE -> Deletes the coupled rows in photos
  ```
  CREATE TABLE photos (
  id SERIAL PRIMARY KEY,
  url VARCHAR(200),
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
  );
  ```
  - ON DELETE SET NULL -> corresponding user_id of photos becomes NULL
  ```
  CREATE TABLE photos (
  id SERIAL PRIMARY KEY,
  url VARCHAR(200),
  user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
  );
  ```
  - ON DELETE SET DEFAULT -> corresponding user_id of photos becomes  a default value, when provided

## 37. Commands for testing
```
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(50)
  );
INSERT INTO users (username)
VALUES ('john'), ('Mia'), ('Loui'), ('Coco');

CREATE TABLE photos (
id SERIAL PRIMARY KEY,
url VARCHAR(200),
user_id INTEGER REFERENCES users(id)
);

INSERT INTO photos (url, user_id)
VALUES
('http:/one.jpg', 4),
('http:/two.jpg', 1),
('http:/25.jpg', 1),
('http:/36.jpg', 1),
('http:/754.jpg', 2),
('http:/35.jpg', 3),
('http:/256.jpg', 4);
```

## 41. Adding some data
```
CREATE TABLE users(
  id SERIAL PRIMARY KEY,
  username VARCHAR(50)
);

CREATE TABLE photos (
  id SERIAL PRIMARY KEY,
  url VARCHAR(200),
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE comments (
  id SERIAL PRIMARY KEY,
  contents VARCHAR(240),
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  photo_id INTEGER REFERENCES photos(id) ON DELETE CASCADE
);

INSERT INTO users (username)
VALUES
  ('Reyna.Marvin'),
        ('Micah.Cremin'),
        ('Alfredo66'),
        ('Gerard_Mitchell42'),
        ('Frederique_Donnelly');

INSERT INTO photos (url, user_id)
VALUES
  ('https://santina.net', 3),
        ('https://alayna.net', 5),
        ('https://kailyn.name', 3),
        ('http://marjolaine.name', 1),
        ('http://chet.net', 5),
        ('http://jerrold.org', 2),
        ('https://meredith.net', 4),
        ('http://isaias.net', 4),
        ('http://dayne.com', 4),
        ('http://colten.net', 2),
        ('https://adelbert.biz', 5),
        ('http://kolby.org', 1),
        ('https://deon.biz', 2),
        ('https://marina.com', 5),
        ('http://johnson.info', 1),
        ('https://linda.info', 2),
        ('https://tyrique.info', 4),
        ('http://buddy.info', 5),
        ('https://elinore.name', 2),
        ('http://sasha.com', 3);

INSERT INTO comments (contents, user_id, photo_id)
VALUES
  ('Quo velit iusto ducimus quos a incidunt nesciunt facilis.', 2, 4),
        ('Non est totam.', 5, 5),
        ('Fuga et iste beatae.', 3, 3),
        ('Molestias tempore est.', 1, 5),
        ('Est voluptatum voluptatem voluptatem est ullam quod quia in.', 1, 5),
        ('Aut et similique porro ullam.', 1, 3),
        ('Fugiat cupiditate consequatur sit magni at non ad omnis.', 1, 2),
        ('Accusantium illo maiores et sed maiores quod natus.', 2, 5),
        ('Perferendis cumque eligendi.', 1, 2),
        ('Nihil quo voluptatem placeat.', 5, 5),
        ('Rerum dolor sunt sint.', 5, 2),
        ('Id corrupti tenetur similique reprehenderit qui sint qui nulla tenetur.', 2, 1),
        ('Maiores quo quia.', 1, 5),
        ('Culpa perferendis qui perferendis eligendi officia neque ex.', 1, 4),
        ('Reprehenderit voluptates rerum qui veritatis ut.', 1, 1),
        ('Aut ipsum porro deserunt maiores sit.', 5, 3),
        ('Aut qui eum eos soluta pariatur.', 1, 1),
        ('Praesentium tempora rerum necessitatibus aut.', 4, 3),
        ('Magni error voluptas veniam ipsum enim.', 4, 2),
        ('Et maiores libero quod aliquam sit voluptas.', 2, 3),
        ('Eius ab occaecati quae eos aut enim rem.', 5, 4),
        ('Et sit occaecati.', 4, 3),
        ('Illum omnis et excepturi totam eum omnis.', 1, 5),
        ('Nemo nihil rerum alias vel.', 5, 1),
        ('Voluptas ab eius.', 5, 1),
        ('Dolor soluta quisquam voluptatibus delectus.', 3, 5),
        ('Consequatur neque beatae.', 4, 5),
        ('Aliquid vel voluptatem.', 4, 5),
        ('Maiores nulla ea non autem.', 4, 5),
        ('Enim doloremque delectus.', 1, 4),
        ('Facere vel assumenda.', 2, 5),
        ('Fugiat dignissimos dolorum iusto fugit voluptas et.', 2, 1),
        ('Sed cumque in et.', 1, 3),
        ('Doloribus temporibus hic eveniet temporibus corrupti et voluptatem et sint.', 5, 4),
        ('Quia dolorem officia explicabo quae.', 3, 1),
        ('Ullam ad laborum totam veniam.', 1, 2),
        ('Et rerum voluptas et corporis rem in hic.', 2, 3),
        ('Tempora quas facere.', 3, 1),
        ('Rem autem corporis earum necessitatibus dolores explicabo iste quo.', 5, 5),
        ('Animi aperiam repellendus in aut eum consequatur quos.', 1, 2),
        ('Enim esse magni.', 4, 3),
        ('Saepe cumque qui pariatur.', 4, 4),
        ('Sit dolorem ipsam nisi.', 4, 1),
        ('Dolorem veniam nisi quidem.', 2, 5),
        ('Porro illum perferendis nemo libero voluptatibus vel.', 3, 3),
        ('Dicta enim rerum culpa a quo molestiae nam repudiandae at.', 2, 4),
        ('Consequatur magnam autem voluptas deserunt.', 5, 1),
        ('Incidunt cum delectus sunt tenetur et.', 4, 3),
        ('Non vel eveniet sed molestiae tempora.', 2, 1),
        ('Ad placeat repellat et veniam ea asperiores.', 5, 1),
        ('Eum aut magni sint.', 3, 1),
        ('Aperiam voluptates quis velit explicabo ipsam vero eum.', 1, 3),
        ('Error nesciunt blanditiis quae quis et tempora velit repellat sint.', 2, 4),
        ('Blanditiis saepe dolorem enim eos sed ea.', 1, 2),
        ('Ab veritatis est.', 2, 2),
        ('Vitae voluptatem voluptates vel nam.', 3, 1),
        ('Neque aspernatur est non ad vitae nisi ut nobis enim.', 4, 3),
        ('Debitis ut amet.', 4, 2),
        ('Pariatur beatae nihil cum molestiae provident vel.', 4, 4),
        ('Aperiam sunt aliquam illum impedit.', 1, 4),
        ('Aut laudantium necessitatibus harum eaque.', 5, 3),
        ('Debitis voluptatum nesciunt quisquam voluptatibus fugiat nostrum sed dolore quasi.', 3, 2),
        ('Praesentium velit voluptatem distinctio ut voluptatum at aut.', 2, 2),
        ('Voluptates nihil voluptatum quia maiores dolorum molestias occaecati.', 1, 4),
        ('Quisquam modi labore.', 3, 2),
        ('Fugit quia perferendis magni doloremque dicta officia dignissimos ut necessitatibus.', 1, 4),
        ('Tempora ipsam aut placeat ducimus ut exercitationem quis provident.', 5, 3),
        ('Expedita ducimus cum quibusdam.', 5, 1),
        ('In voluptates doloribus aut ut libero possimus adipisci iste.', 3, 2),
        ('Sit qui est sed accusantium quidem id voluptatum id.', 1, 5),
        ('Libero eius quo consequatur laudantium reiciendis reiciendis aliquid nemo.', 1, 2),
        ('Officia qui reprehenderit ut accusamus qui voluptatum at.', 2, 2),
        ('Ad similique quo.', 4, 1),
        ('Commodi culpa aut nobis qui illum deserunt reiciendis.', 2, 3),
        ('Tenetur quam aut rerum doloribus est ipsa autem.', 4, 2),
        ('Est accusamus aut nisi sit aut id non natus assumenda.', 2, 4),
        ('Et sit et vel quos recusandae quo qui.', 1, 3),
        ('Velit nihil voluptatem et sed.', 4, 4),
        ('Sunt vitae expedita fugiat occaecati.', 1, 3),
        ('Consequatur quod et ipsam in dolorem.', 4, 2),
        ('Magnam voluptatum molestias vitae voluptatibus beatae nostrum sunt.', 3, 5),
        ('Alias praesentium ut voluptatem alias praesentium tempora voluptas debitis.', 2, 5),
        ('Ipsam cumque aut consectetur mollitia vel quod voluptates provident suscipit.', 3, 5),
        ('Ad dignissimos quia aut commodi vel ut nisi.', 3, 3),
        ('Fugit ut architecto doloremque neque quis.', 4, 5),
        ('Repudiandae et voluptas aut in excepturi.', 5, 3),
        ('Aperiam voluptatem animi.', 5, 1),
        ('Et mollitia vel soluta fugiat.', 4, 1),
        ('Ut nemo voluptas voluptatem voluptas.', 5, 2),
        ('At aut quidem voluptatibus rem.', 5, 1),
        ('Temporibus voluptates iure fuga alias minus eius.', 2, 3),
        ('Non autem laboriosam consectetur officiis aut excepturi nobis commodi.', 4, 3),
        ('Esse voluptatem sed deserunt ipsum eaque maxime rerum qui.', 5, 5),
        ('Debitis ipsam ut pariatur molestiae ut qui aut reiciendis.', 4, 4),
        ('Illo atque nihil et quod consequatur neque pariatur delectus.', 3, 3),
        ('Qui et hic accusantium odio quis necessitatibus et magni.', 4, 2),
        ('Debitis repellendus inventore omnis est facere aliquam.', 3, 3),
        ('Occaecati eos possimus deleniti itaque aliquam accusamus.', 3, 4),
        ('Molestiae officia architecto eius nesciunt.', 5, 4),
        ('Minima dolorem reiciendis excepturi culpa sapiente eos deserunt ut.', 3, 3);
```

## 42. JOINS and AGGREGATION
- JOINS: merging rows and process data
- AGGREGATIONS: Collective operation such as most, average, least, ...

## 43. JOINING data from other tables
```
SELECT contents, username, users.id, user_id
FROM comments
JOIN users ON users.id = comments.user_id;
```
- When key names are ambiguous (or duplicating) like `id`, use table name with like `users.id`

## 47. More on JOINS
- Using AS
```
SELECT contents, username, users.id, user_id
FROM comments AS c
JOIN users ON users.id = c.user_id;
```

## 48. Missing data in JOINS
- When photos.user_id is NULL
  - The above command will not report the case of NULL id
  - FULL or LEFT or RIGHT JOIN is necessary for mis-matching or NULL cases

## 50. 4 kinds of JOIN
- INNER JOIN
  - Regular `JOIN` implies `INNER JOIN`
  - [photos [] users]
```
SELECT url, username
FROM photos
INNER JOIN users ON users.id = photos.user_id;
```
- LEFT JOIN
  - Same as LEFT OUTER JOIN
  - [[photos] users]
  - When values are not found, NULL is printed
```
SELECT url, username
FROM photos
LEFT JOIN users ON users.id = photos.user_id;
```
- RIGHT JOIN
  - Same as RIGHT OUTER JOIN
  - [photos [users]]
```
SELECT url, username
FROM photos
RIGHT JOIN users ON users.id = photos.user_id;
```
- FULL JOIN
  - Same as FULL OUTER JOIN
  - [[photos users]]
```
SELECT url, username
FROM photos
FULL JOIN users ON users.id = photos.user_id;
```

## 55. WHERE with JOIN
```
SELECT url, contents
FROM COMMENTS
JOIN photos ON photos.id = comments.photo_id
WHERE photos.user_id = comments.user_id;
```

## 56. Multiple JOINs
- Merging tables of comments, photos, and users
```
SELECT url, contents, username
FROM COMMENTS
JOIN photos ON photos.id = comments.photo_id
JOIN users ON users.id = comments.user_id
AND users.id = photos.user_id;
```

## 59. Aggregating and grouping
