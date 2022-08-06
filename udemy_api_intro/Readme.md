## Udemy API and Web Service Introduction
- Nate Ross

## Section 1: Introduction

1. Introduction

## Section 2: API

2. What is an API?
  - Application: Does an actual task
  - Programming: Does the taks in the Application
  - Interface: Tells the Program to run
  - Program independent
  - Use program without writing

3. API Exercise 1

4. API Exercise 2
   - Ebay API : https://developer.ebay.com/

5. API Details
  - At every API transaction
    - Request
    - Program
    - Response

6. API Details Example

7. API Mash up

## Section 3: Web Service

8. What is a Web service?
  - All web service is an API
  - Web Services use
    - XML/JSON to format data
    - REST, SOAP to transfer data

9. API and Web Service Conclusion

## Section 4: HTTP

10. Introduction to HTTP
  - Request -> Program -> Response

| HTTP     | Request                             | Response |
|----------|-------------------------------------|---------|
|Start line| version http1.1, Method, parameters |  version http1.1, STATUS code|
| Headers |  Host address, Token, credentials    |  Cookies, size of data|
| Blank line | Just place holder                 | Just place holder |
| Body       | nothing or input data             | HTML         |

11. HTTP Exercise

12. HTTP Parts

13. HTTP Start Line

|       | Request | Response|
|-------|---------|---------|
|Name | Start Line, Request Line | Start Line, Response Line, Status Line |
| HTTP Version | HTTP/1.1  | HTTP/1.1 |
| Method | GET, POST, PUT, DELETE, etc |  No|
| API program folder location | Yes(ex: /search) | No | 
| Parameters| Yes(ex: ?q=tuna) |No|
| Status code | No | Yes (ex: 200 OK) |
| Format | Method + space + API program folder location + parameters + space + HTTP version | HTTP version + status code |
| Example | GET /search?q=tuna HTTP/1.1 | HTTP/1.1 200 OK |

  - HTTP request methods: https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
  - HTTP response status codes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
  - Idempotence: SAFE to repeat
    - GET: Yes. Can repeat as many as you ant
    - POST: No
    - PUT: Yes
    - DELETE: Yes 

14. HTTP Start Line Exercise

15. HTTP Headers
  - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers
  - https://en.wikipedia.org/wiki/List_of_HTTP_header_fields
    - Request field and Response field have different keywords

16. HTTP Headers Exercise

17. HTTP Blank line

18. HTTP Body
  - Content-Type from HTTP headers
  - Typical data format: XML/JSON

19. HTTP Body Exercise

20. Advanced Topics HTTP: Stateless, Cookies, Security, Infrastructure
  - Stateless: Request unknown
    - Ex) regular phone call. You don't know who calls / where it comes from until you talk
    - HTTP is stateless by default
    - HTTP methods:  REST and SOAP
    - REST and SOAP are stateless by default
  - Stateful: File Transfer Protocol (ftp)
    - If the web server stores this data in a backend manner and uses it to identify you as a constantly connected client, the service is stateful
  - How do applications remember?
    - Cookies
    - HTTP header sets cookie
    - Cookies are not executable
    - May store tokens but not passwords
  - Benefit of stateless infrastructure
    - Scalability
    - Resilience 
    - Less memory needed

## Section 5: XML

21. XML Introduction
  - eXtensible Markup Language
  - HTTP Header line: Content type: application/xml
  - HTTP Body: XML
  - Looks similar to HTML
    - Created by W3C

22. XML Exercise

## Section 6: JSON

23. JSON Introduction
  - Java Script Object Notation
  - HTTP Header Line: Content-Type: application/json
  - HTTP Body: JSON
  - Key with Value 
```json
{"Pizza": [ 
    {"Size": "Small", 
     "Toppings": ["Onions", "Mushrooms"]
    }
  ]
}
```

24. JSON Exercise

25. XML JSON Comparison

|            | XML | JSON |
|------------|-----|------|
| Powerful   | Yes | No|
|Simple     |No    | Yes|
| Developed | 1997 | 2001 |
| Popularity| Down | Up |

## Section 7: SOAP

26. SOAP REST Comparison
  - How they different when form HTTP Request and Response

27. SOAP Introduction
  - Simple Object Access Protocol
  - Uses Web Services Description Language (WSDL)
  - 4 HTTP Request parts
    - START Line: POST WSDL HTTP version
    - Header Line: Content-Type: text/xml
    - Blank Line
    - Body: XML envelope formed using WSDL
  - Created by W3C: creator of XML and HTML

28. SOAP Examples
  - API provider will provide corresponding WSDL map

29. SOAP Exercise

## Section 8: REST

30. REST Introduction Part one

|time   | API Method|
|-------|-----------|
| 1980s |   SUN RPC |
| 1998  | XML-RPC   |
| 1999  | SOAP      |
| 2000  | REST      |

  - How HTTP Request/Response works
    - XML-RPC/SOAP: 
      - Interface -> HTTP request with POST -> Program
        - Default METHOD is POST which doesn't do anything
      - Program -> HTTP response -> Interface
    - REST:
      - Interface -> HTTP request with METHOD -> Program
        - In database analogue, CRUD is done
          - Create -> POST
          - Read -> GET
          - Update -> PUT
          - Delete -> DELETE
        - Much lighter than SOAP
      - Program -> HTTP response -> Interface
  
31. REST Introduction Part two
  - Representational State Transfer
  - No rule like WDSL in SOAP
  - 4 HTTP Request parts
    - START Line: All methods (GET/POST/PUT/DELETE/...)
    - Header Line: All
    - Blank Line
    - Body: Any (JSON, XML, images, ...)

32. REST Example (Call EBay API)

33. REST Example (Create Twitter app)

34. REST Exericse

## Section 9: API Access

35. HTTPS

36. Authentication and Authorization
- Authentication: proving your identity through credentials
- Authorization: limited access

| Name |Authentication | Authorization |Examples |
|------|---------------|---------------|---------|
| No auth     | N | N | Google search engine|
| Basic auth  | Y | N | Email account |
|Bearer Token | N | Y | Not many |
| OAuth       | Y | Y | Many smart phone apps |
| Two factor authentication | Y | N | High security of Bank/financial service |

37. Apps

38. OAuth
  - RFC note 6749
  - Roles
    - Resource owner
    - Resource server
    - Client/application
    - Authorization server
  - Protocol flow
    - Client : authorization request -> Resource owner
    - Resource owner : authorization grant -> Client
    - Client : Autherization grant -> Auth server
    - Auth server : Access token -> Client
    - Client : Access token -> Resource server
    - Resource server : Protected resource -> Client
  - Authorization grant type
    - Authorization code: most popular
    - Implicit
    - Resource owner
    - Client credentials
  - Access token: limited time span
  - Refresh token: Will renew access token (?)

39. Advanced Topic: OAuth Authorization Code Grant (Google Cloud)
  - Authorization code grant
    - Client -> User Agent -> client ID + Rediret URL -> Auth server
    - User Agent -> User authentication -> User agent
    - Auth Server -> Authorization code -> User Agent -> Client
    - Client -> Authorization code + Redirect URL -> Auth Server
    - Auth Server -> Access Token (optional Refresh Token) -> Client
  - Client Registration
    - Client HTTP to Auth Server
      - Client type (public/confidential)
      - Redirect URL
      - Client Secret
      - Additional info (name, website, email, ...)
    - Auth Server HTTP to client
      - Client ID
      - Client secret
    - Access Token Request
      - HTTP method may be POST or GET 
        - Sends:
          - Grant type = authorization code
          - Code = the authorization code
          - Client ID
          - Client secret
      - Auth server sends back HTTP to client with:
        - Access token
        - Token type
        - Expires in (seconds)
        - Refresh token (optional)
        - Scope (optional)

40. Advanced Topic: OpenID Connect
  
41. Exercise: API Access Definitions

## Section 10: API Examples

42. Postman Installation
  - https://www.postman.com/downloads/

43. Postman Simple Example
  - Postman is a browser
  - Can call an API without the need of an interface
  - Try get with `www.ebay.com` then click Send
    - Check the response
    - 2845 lines of Body
    - 28 Headers
    - 12 Cookies 

44. No Authentication
  - Sample No Auth API: https://mixedanalytics.com/blog/list-actually-free-open-no-auth-needed-apis/
  - Let's use https://archive.readme.io/docs/item
    - Get `https://archive.org/metadata/principleofrelat00eins`

45. Basic Authentication
  - Acccess an API using username and passwd
  - Not very secure as credentials can be intercepted
    - Use OAuth
  - Practice:
    - In the Postman, select `Basic Auth`

46. Digest Authentication
  - A way of providing your identity by using a digested crendential
  - Not used often

47. Bearer Token
  - In 2012 as a part of the OAuth specification
  - httpbin.org: can test various auth methods

48. OAuth
  - Delegates limited authorization to an application
  - App uses access token
  - Started in 2006
  - Very popular
  - OAuth play ground by Google
    - Gmail API v1

49. Postman Exercise

50. Python

## Section 11: Extras

51. Create an API (AWS)

| API  protocol | Unsecure | Secure |
|---------------|----------|--------|
| HTTP          | HTTP     |  HTTPS |
| Web Socket    | WS       |   WSS  |


52. Calling APIs using programming languages
  - Postman can convert the API call into specific language version
```python
import requests
url = "https://archive.org/metadata/principleofrelat00eins?uniq=827189196&size=11635"
payload={}
headers = {}
response = requests.request("GET", url, headers=headers, data=payload)
print(response.text)
```

53. Webhooks
  - Reverse API
  - Steps
    - Bank has a table of IP address vs Phone number
    - Event/Trigger
    - Endpoint/Configuration to login email account/contact cellphone via HTTP Request
    - HTTP response to the Bank

54. Microservices
  - Microservices vs Monolith architecture
  - Monolith
    - One API
  - Microservices
    - Collection of many APIs
    - Advantages:
      - Scalability
      - Code language independence
      - Smaller and specialized team
    - Disadvantage
      - Lack of consistency      

55. What's next/after this course

## Section 12: Conclusion

56. Conclusion
