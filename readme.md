# This repository contains client and server side of TRLV (the risen Lord's vineyeard website)

# requirements:
1. git
2. conda
3. python

# Source code usage
1. assuming git is installed clone repository by running `git clone https://github.com/08Aristodemus24/project-trlv`
2. assuming conda is also installed run `conda create -n project-trlv python=3.12.3`. Note python version should be `3.12.3` for the to be created conda environment to avoid dependency/package incompatibility.
3. run `conda activate project-trlv` or `activate project-trlv`.
4. run `conda list -e` to see list of installed packages. If pip is not yet installed run conda install pip, otherwise skip this step and move to step 5.
5. navigate to directory containing the `requirements.txt` file.
5. run `pip install -r requirements.txt` inside the directory containing the `requirements.txt` file
6. after installing packages/dependencies run `python index.py` while in this directory to run app locally

# App usage:
1. control panel of app will have 3 inputs: prompt, temperature, and sequence length. Prompt can be understood as the starting point in which our model will append certain words during generation for instance if the prompt given is "jordan" then model might generate "jordan is a country in the middle east" and so on. Temperature input can be understood as "how much the do you want the model to generate diverse sequences or words?" e.g. if a diversity of 2 (this is the max value for diversity/temperature by the way) then then the model might potentially generate incomprehensible words (almost made up words) e.g. "jordan djanna sounlava kianpo". And lastly Sequence Length is how long do you want the generated sequence to be in terms of character length for isntance if sequence length is 10 then generated sequence would be "jordan is."

# File structure:
```
|- client-side
    |- public
    |- src
        |- assets
            |- mediafiles
        |- boards
            |- *.png/jpg/jpeg/gig
        |- components
            |- *.svelte/jsx
        |- App.svelte/jsx
        |- index.css
        |- main.js
        |- vite-env.d.ts
    |- index.html
    |- package.json
    |- package-lock.json
    |- ...
|- server_side
    |- modelling
        |- data
        |- figures & images
            |- *.png/jpg/jpeg/gif
        |- final
            |- misc
            |- models
            |- weights
        |- metrics
            |- __init__.py
            |- custom.py
        |- models
            |- __init__.py
            |- arcs.py
        |- research papers & articles
            |- *.pdf
        |- saved
            |- misc
            |- models
            |- weights
        |- utilities
            |- __init__.py
            |- loaders.py
            |- preprocessors.py
            |- visualizers.py
        |- __init__.py
        |- experimentation.ipynb
        |- testing.ipynb
        |- training.ipynb
    |- static
        |- assets
            |- *.js
            |- *.css
        |- index.html
    |- index.py
    |- server.py
    |- requirements.txt
|- demo-video.mp5
|- .gitignore
|- readme.md
```

# Articles:
1. 

# Insights
## Typescript
* types in typescript: `number`, `string`, `boolean`, `any` encompasses any primitive type like number, string, boolean,  `any[]`, <primitive type>[] defines an array of number

## JWT Authentication
* The code you provided doesn't run constantly in the background to proactively refresh the token before it expires. Instead, it takes a reactive approach to token expiration.

Here's how it works and why it might seem like it's constantly checking (but isn't):

The useEffect Hook Trigger: The auth() function is called only once when the ProtectedRoute component initially mounts because of the empty dependency array [] in the useEffect hook.

Checking on Route Access: The auth() function is executed every time a user tries to access a route wrapped by this ProtectedRoute component. This is because the ProtectedRoute component itself is rendered each time the route changes.

Expiration Check within auth(): Inside the auth() function, the code checks if the accessToken in localStorage exists and if it's expired:

JavaScript

if(tokenExpiration < now){
    await refreshToken();
}
This check happens only when the auth() function is called (i.e., when trying to access a protected route).

Reactive Refresh: If the accessToken is found to be expired at the moment the user tries to access a protected route, the refreshToken() function is called to get a new one.

Why this isn't a constant background process:

Resource Efficiency: Running a background process that constantly checks the token expiration and makes API calls would be inefficient and consume unnecessary resources on the user's browser.
Event-Driven: The need to check authentication and potentially refresh the token is primarily driven by user actions (trying to access a protected resource).
How to achieve more proactive refreshing (and why the provided code might not be ideal for that):

The provided code has a potential drawback: if the access token expires while the user is on a protected page and they try to perform an action that requires authentication, the API request will likely fail first. Only then, on the next attempt to access a protected route (which might be triggered by a redirect or another user action), will the auth() function run and attempt to refresh the token.

To implement more proactive refreshing, you would typically employ one of these strategies:

Using a Timer: You could set a timer after a successful login that triggers a token refresh a certain amount of time before the access token is expected to expire. This would require more complex logic to manage the timer and ensure it's cleared when the user logs out or navigates away from protected areas.
Intercepting API Requests: A more common and often cleaner approach is to use an HTTP interceptor (provided by libraries like Axios) to automatically check for token expiration before making an authenticated API request. If the token is expired or close to expiring, the interceptor can first attempt to refresh the token and then retry the original API request. This makes the token refresh process more seamless for the user.
Background Tasks (Service Workers - More Advanced): For more complex scenarios, you could potentially use Service Workers to perform background tasks, including token refresh. However, this adds significant complexity to your application.
In the context of the provided code:

The code you shared focuses on ensuring that when a protected route is accessed, the authentication status is checked, and a refresh is attempted if the access token has already expired. It doesn't actively try to refresh the token before it expires without the user initiating a navigation to a protected route.

While this reactive approach works to protect routes, it might lead to a slightly less smooth user experience if API calls fail due to an expired token before the next route change triggers the refresh.

Therefore, while it doesn't "run constantly," the auth() function is invoked each time a protected route is rendered, which effectively checks the token's validity at that point. For a more proactive approach, you'd need to add more sophisticated logic, often involving timers or HTTP request interception.

* 
```
curl \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"username": "davidattenborough", "password": "boatymcboatface"}' \
  http://localhost:8000/api/token/

...
{
  "access":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX3BrIjoxLCJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiY29sZF9zdHVmZiI6IuKYgyIsImV4cCI6MTIzNDU2LCJqdGkiOiJmZDJmOWQ1ZTFhN2M0MmU4OTQ5MzVlMzYyYmNhOGJjYSJ9.NHlztMGER7UADHZJlxNG0WSi22a2KaYSfd1S-AuT7lU",
  "refresh":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX3BrIjoxLCJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImNvbGRfc3R1ZmYiOiLimIMiLCJleHAiOjIzNDU2NywianRpIjoiZGUxMmY0ZTY3MDY4NDI3ODg5ZjE1YWMyNzcwZGEwNTEifQ.aEoAYkSJjoWH1boshQAaTkf8G3yn0kapko6HFRt7Rh4"
}
```

* 
in python
```
request.get("http://localhost:8000/api/some-protected-view/", headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX3BrIjoxLCJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiY29sZF9zdHVmZiI6IuKYgyIsImV4cCI6MTIzNDU2LCJqdGkiOiJmZDJmOWQ1ZTFhN2M0MmU4OTQ5MzVlMzYyYmNhOGJjYSJ9.NHlztMGER7UADHZJlxNG0WSi22a2KaYSfd1S-AuT7lU"})
```

in javascript
```
fetch("http://localhost:8000/api/some-protected-view/", {
    method: "POST",
    headers: {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX3BrIjoxLCJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiY29sZF9zdHVmZiI6IuKYgyIsImV4cCI6MTIzNDU2LCJqdGkiOiJmZDJmOWQ1ZTFhN2M0MmU4OTQ5MzVlMzYyYmNhOGJjYSJ9.NHlztMGER7UADHZJlxNG0WSi22a2KaYSfd1S-AuT7lU",
        "Content-Type": "application/json",
    },
    body: JSON.stringify({
        "<field 1>": "<field 1 value>",
        "<field 2>": "<field 2 value>",
        ...
        "<field n>": "<field n value>"
    })
})
```