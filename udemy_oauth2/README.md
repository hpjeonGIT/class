## The Nuts and Bolts of OAuth 2.0
- Instructor: Aaron Parecki

2. History
- Early 2000: Facebook asked gmail credential to login
- Mid 2000: without sharing passwd, third party allowed to login. fragmented approach by each vendor
- 2007: first Oauth
- Development: Separating auth server and application server
- Oauth was created as a delegated authorization protocol. It has been extended to be used as a single sign-on protocol through extra stuffs like OpenID Connect.

3. How Oauth improves application security
- Using session cookie: saving credential at local system
    - Each app needs separate cookies
- How to use one auth for multiple applications?

4. Oauth vs OpenID
- Oauth: was developed for application to access API
    - Oauth server: gives you a keycard
    - Application server: uses the keycard to allow you. You don't deliver send your credential to application server.
    - Doesn't give ID of users to applications
    - Applications don't parse the access tokens. Just let API figure out if the token is valid or not.
- OpenID: can provide extra credential on top of Oauth, identifying users.
    - Provides ID of users to applications
    - ID token

5. Roles in Oauth
- Users: resource owner
- Device : browser that accesses application. User agent. 
- Application: oauth client. Never sees the user's credential
- API: resource server
- Authorization server: 

6. Application types
- Confidential: contains secret
- Public: doesn't contain secret - source is visible in mobile apps.
- Authorization server may not know the request from application is valid or not
    - Credentialed client: 
- Never put client secret in mobile app or single page app (SPA)

7. User consent

8. Front vs back channel
- Back channel: secure way like https. Certificate enscription. Trusted response. A request from an http client to an http server
- Front channel: moving browser address b/w application. Can be snatched. cannot be sure if the package is fake or not.
    - Sending data by making the browser redirect
- Implicit flow: in old ways. not recommended. 
    - CORS (Cross Origin Resource Sharing) is available now

9. Application identity
- Each application has client ID
- Authorization code issued from authorization server
- Client secret
- PKCE: 
- Redirect URI: where authorization code is sent in front channel

Quiz2:
- When a user is using AAA CI to test code from GitHub. 
    - GitHub is playing the role of: Authorization server and resource server
    - AAA CI is: client
- If a user is developing an application with .NET backend, this application is: confidential, as it is a server-side.
    - Mobile apps correspond to be public

10. Oauth client

11. Oauth server side - registering an application
- Client secret shouldn't be included  in the app

12. Authorization code flow for web application
- PKCE code verifier: when a user requests to use an app, the app generates a new secret/hashes it -> the user contacts the authorization server with the hashed secret (this is done through front channel) -> Authorization server generates a temporary code to the user, which is one time use and short-lived. Then the user forwards the temporary code to the app got get a token (this is done through another front channel) -> Application contacts authorization server with the delivered temporary code, which is PKCE.
    - PKCE may prevent authorization code inject

13. Oauth for native application - unique issues
- no client secret in the mobile app!

14. Redirect URLs for native apps
- In mobile app, redirecting can be snatched
- Different apps may use the same url to invoke

15. Browser security for native apps
- Each mobile app opens a browser
- No shared cookies b/w mobile apps

16. Authorization code flow for native apps

17. Refresh tokens for native apps
- apps cannot see refresh tokens
- Based on the stored refresh token and local credential (finger print?), new token can be downloaded

18. Problems with the browser environment
- SPA: gmail, google map, ... doesn't require reloading when a single user uses it
- MPA: amazon, ebay, ... many pages are refreshed whenever data chage
- XSS (Cross Site Script) attack: strong contents policy may defend but not complete
- Be aware of potential vulnerability

19. Authorization code flow for SPA
- Users -> App. PCKE in the app: will generate hash code. Take this has code with you to authorization server -> done by front channel -> login by user -> Authorization server generates a temporary code that the app can use. Authorization server doesn't know if the the temporary code is actually received by the correct app -> Users use the temporary code to get a token (front channel) -> App sends the plaintext secret to Authorization server, asking a token (back channel) -> Authorization Server verifies the has of the secret, and gives a token when valid (back channel)
    
20. Protecting tokens in the browser
- Main storage of javascript
    - Local storage
    - Session storage: session data is not shared by tabs
    - Cookies: 
    - All vulnerable to XSS:
        - Keep tokens in memory: hard to use in other apps
        - Keep tokens in service worker:  
        - Webcrypto: not common yet

21. Securing the browser with a backend
- Token out of JS apps
- Dynamic backend server
    - Will not work AWS + S3
    - But might be good for SPA

22. Limitations of IoT and smartphone devices
- Instead of logging from the console, let users login from computer. Then device will gain the access
- Device flow is quite different thant mobile app

23. Flow in device
- Users -> devices -> Authorization Server -> sends temporary code and URL info -> Device : shows the URL and info to users -> Users: login to URL and access -> Authorization Server: it knows that users logged in. Device->AS : has the user logged in? AS: got token -> Device: will let user access

24. When to use the client credentials grant

26. OpenID token
- Extends Oauth
- Applications learn about users
- ID token: json web token
    - header/payload/signature

27. How ID tokens are different from access tokens
 - They look similar as json
 - Access token is used by Application to access API. No user info
 - ID tokens contain User info

 30. Validating and using an ID token
 - Validate signature

 32. Reference tokens vs Self-encoded tokens
 - Access token could be reference or structured token
    - Reference token:
    - Structured token:

33. Pros and cons of reference tokens
- It is hidden from applications. Only open to authorization server
- But not stored

34. Self encoded tokens
- Is structured
- Doesn't need shared storage
- Can be validated without network
- Scalable
- Jason Web Token (JWT) is visible
- No way to revoke (until expiration)

35. The structure of JWT access token
- iss, exp, iat, aud, sub, client_id, jti, scope, auth_name, acr, amr

36. Remote token introspection

37. Local token validation
- Use oauth libs
- avoid `alg: none`

38. The best of both worlds: Using an API gateway
- Local validation + introspection?
- API gateway: only local validation
- Backend API may handle introspection

39. Short Token lifetime

46. The purpose of OAuth scopes
- Request of access
- Limits what applications do with API

47. Defining scopes for your API
