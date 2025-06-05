# CAIC Summer of Tech 2025: Full-Stack Chat App Development

## Week 1: Foundation Setup and Learning Prerequisites

Welcome to Week 1 of the Full-Stack Development Track! This foundational week is crucial for setting up your development environment and understanding the core concepts that will be essential throughout our 5-week journey. Before we dive into building our messaging app, we need to establish a solid understanding of web technologies, development tools, and the fundamental concepts that separate backend and frontend development.

**What is Backend & Frontend and How it works?** → [How The Backend Works](https://www.youtube.com/watch?v=4r6WdaY3SOA)

## Learning Tasks

So, let's get started with learning first of all... Read and Digest the following resources before moving forward to the Implementation Tasks.

### Understanding HTTP and Web Communication

Each request sent from the client's side results in a response from the server which has a few components:

**Status Line:**
- **HTTP Version:** Indicates the HTTP version used, e.g., HTTP/1.1, HTTP/2
- **Status Code:** A three-digit code indicating the result of the request:
  - 200: OK (Success)
  - 404: Not Found (Resource doesn't exist)
  - 500: Internal Server Error (Server-side problem)
  - [Mozilla's HTTP Status Codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
- **Reason Phrase:** A brief description of the status code, e.g., OK, Not Found, Internal Server Error

**Headers:**
- Contain metadata about the request/response
- Include information like content type, authentication tokens, caching instructions
- [Mozilla's HTTP Headers Documentation](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers)

**Body:**
- Contains the actual data requested by the client
- Can be in various formats:
  - **JSON:** Most commonly used in mobile APIs for structured data
  - **HTML:** For web pages (less relevant for mobile apps)
  - **Binary data:** Such as images, videos, or files
  - **XML:** Another format for structured data

**Optional Components:**
- **Cookies:** Data sent by the server to be stored on the client-side
- [CloudFlare's HTTP Cookies Guide](https://www.cloudflare.com/learning/privacy/what-are-cookies/)

### Additional Essential Concepts

**Browser Developer Tools**

Master Chrome DevTools:
- Elements tab for HTML/CSS inspection
- Console tab for JavaScript debugging
- Network tab for monitoring API requests
- Application tab for viewing local storage
- [Chrome DevTools Documentation - Browser debugging](https://developer.chrome.com/docs/devtools/)
- Resource: [Inspect Network Activity - Chrome DevTools 101](https://developer.chrome.com/docs/devtools/network/)

**API Testing Tools:**
- **Postman:** Industry-standard tool for testing APIs
- [Learn Postman in 15 Minutes :)](https://www.youtube.com/watch?v=VywxIQ2ZXw4)

## Week 1 Implementation Tasks

### Task 1: Development Environment Setup

#### Task 1A: Backend Environment Setup (Node.js)

**Install Node.js (Latest LTS version recommended)**

- Download from [nodejs.org](https://nodejs.org/)
- Verify installation: `node --version` and `npm --version`

**Install Essential Backend Packages:**

```bash
npm init -y
npm install express cors dotenv jsonwebtoken bcryptjs
```

**Set up project structure:**

```
backend/
├── server.js
├── routes/
├── models/
├── middleware/
└── config/
```

[Node.js Getting Started Guide](https://nodejs.org/en/docs/guides/getting-started-guide/)

#### Task 1B: Frontend Environment Setup (React)

**Install Create React App:**

```bash
npx create-react-app my-chat-app
cd my-chat-app
```

**Install Essential Frontend Packages:**

```bash
npm install axios socket.io-client
```

**Set up project structure:**

```
frontend/
├── public/
├── src/
│   ├── components/
│   ├── pages/
│   ├── utils/
│   ├── App.js
│   └── index.js
└── package.json
```

**Start the development server:**

```bash
npm start
```

### Task 2: Core JavaScript Learning

#### Task 2A: Modern JavaScript Fundamentals

Master these essential JavaScript concepts that you'll use daily:
- **ES6+ Features:** Arrow functions, destructuring, template literals, async/await
- **Promises and Async Programming:** Understanding asynchronous operations
- **Array Methods:** map(), filter(), reduce(), forEach()
- **Object Manipulation:** Object.keys(), Object.values(), spread operator
- **Module Import/Export:** ES6 modules for organizing code

**Learning Resources:**
- [JavaScript.info - Modern JavaScript Tutorial](https://javascript.info/)
- [MDN JavaScript Guide](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)

#### Task 2B: API Concepts and Practice

- **Understanding REST APIs:** GET, POST, PUT, DELETE methods
- **JSON Data Format:** Parsing and stringifying JSON data
- **Fetch API:** Making HTTP requests from JavaScript
- **Error Handling:** Try-catch blocks and proper error management

**Practice Exercise:** Use the JSONPlaceholder API to practice making requests:

```javascript
// Example API call
fetch('https://jsonplaceholder.typicode.com/posts')
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error('Error:', error));
```

### Task 3: React Fundamentals

Learn these fundamental React concepts:

- **Components:** Functional components and JSX syntax
- **Props:** Passing data between components
- **State:** Managing component state with useState hook
- **Effects:** Side effects with useEffect hook
- **Event Handling:** Handling user interactions
- **Conditional Rendering:** Showing/hiding elements based on state
- **Component Lifecycle:** Understanding when components mount, update, and unmount

**Learning Resources:**

- [React Official Tutorial](https://reactjs.org/tutorial/tutorial.html)
- [React Hooks Documentation](https://reactjs.org/docs/hooks-intro.html)

### Task 4: Development Tools Mastery

- Install Postman from [postman.com](https://www.postman.com/)
- [Postman Learning Center - API testing tutorials](https://learning.postman.com/)
- Learn basic operations:
  - Creating GET, POST, PUT, and DELETE requests
  - Setting headers and authentication
  - Organizing requests in collections
  - Testing API responses
- Practice with public APIs: JSONPlaceholder, OpenWeatherMap, etc.

## Extra Resources and Documentation

### JavaScript and Web Development
- [MDN Web Docs](https://developer.mozilla.org/) - Comprehensive web development reference
- [JavaScript.info](https://javascript.info/) - Modern JavaScript tutorial

## Testing/Troubleshooting Your Setup

### Verification Checklist
- [ ] Node.js and npm installed and working
- [ ] Basic React setup done.
- [ ] Postman installed and tested with a public API
- [ ] Chrome DevTools explored and understood
- [ ] Basic JavaScript concepts practiced

### Simple Test Project
Create a simple "Hello World" React app that displays a welcome message.
This would confirm that your test environment is working properly :)

## Conclusion

Week 1 is your foundation for success in the remaining 4 weeks. Take time to thoroughly understand these concepts, as they will be essential for building our messaging app. Don't rush through the learning materials – a solid understanding now will make the implementation weeks much smoother.

Remember, full-stack development requires understanding both frontend (what users see) and backend (what powers the app behind the scenes). This week gives you the tools and knowledge to work effectively with both sides of the application.

Next week, we'll start building the core messaging interface and real-time communication features. Make sure you're comfortable with the concepts covered this week before moving forward.

Good luck, and happy coding!
