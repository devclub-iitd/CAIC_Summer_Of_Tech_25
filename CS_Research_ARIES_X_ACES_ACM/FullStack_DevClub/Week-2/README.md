# CAIC Summer of Tech 2025: Full-Stack Chat App Development
## Week 2: Authentication System and Core Messaging Logic

Welcome to Week 2 of the Full-Stack Development Track! This week builds upon the foundation you established in Week 1. Now that you have your development environment set up and understand the basics of HTTP, APIs, and JavaScript, we're ready to start building the core functionality of our messaging application.

This week focuses on creating a secure authentication system using Express + MongoDB for user management, some UI/UX setup- giving you hands-on experience with development concepts.

## What You'll Build This Week

By the end of Week 2, you'll have:
- Backend setup: Node.js and Express server
- Database setup: MongoDB for user data storage
- User models & CRUD routes: Complete user management system
- Basic messaging UI/UX: Clean React interface

## Quick Review: What We Learned in Week 1

- Backend vs Frontend: Backend = server magic, Frontend = user interface
- HTTP Status Codes: 200 (OK), 404 (Not Found), 500 (Server Error)
- Development Tools: Node.js, React, Postman, Chrome DevTools
- Basic JavaScript: Functions, promises, API calls
- Project Structure: Organized folder structure for both frontend and backend

## Learning Tasks

Before diving into implementation, you need to understand these core concepts that will be essential for building our chat application.

### Understanding Authentication Systems

Authentication is crucial for any messaging app - we need to know who's sending messages!

**Essential Video Tutorials:**
- [Authentication Explained in 5 Minutes](https://www.youtube.com/watch?v=YX0gFIjAOLU) - Quick overview
- [JWT Tokens Explained Simply](https://www.youtube.com/watch?v=7Q17ubqLfaM) - 15 min tutorial

**Essential Reading:**
- [What is Authentication? - Auth0](https://auth0.com/intro-to-iam/what-is-authentication) - Simple explanation
- [JWT Introduction](https://jwt.io/introduction) - Read the "What is JSON Web Token?" section

### Backend Setup with Node.js & Express

Time to build the server that will power our chat application!

**Video Tutorials:**
- [Node.js + Express Setup](https://www.youtube.com/watch?v=pKd0Rpw7O48) - 30 min tutorial
- [Building REST APIs with Express](https://www.youtube.com/watch?v=l8WPWK9mS5M) - Complete guide

**Essential Reading:**
- [Express.js Getting Started](https://expressjs.com/en/starter/installing.html) - Official guide
- [RESTful API Design](https://restfulapi.net/) - API design principles

### Database Setup with MongoDB

- [MongoDB Basics](https://www.youtube.com/watch?v=pWbMrx5rVBE) - First 20 minutes
- [Mongoose Getting Started](https://mongoosejs.com/docs/index.html) - Read "Getting Started" section
- [Mongoose Schemas](https://mongoosejs.com/docs/guide.html) - Understanding data models

**Essential Database Concepts:**

**Create Operations:**
- Register a new user account
- Hash passwords before storing

**Read Operations:**
- User login with credential verification
- Retrieve user profile information

**Update Operations:**
- Update user profile details
- Change user password

**Delete Operations:**
- Delete user account from database

### RESTful API Design

**YouTube Tutorial:**
- [REST API Tutorial - What is a RESTful API?](https://www.youtube.com/watch?v=SLwpqD8n3d0) - 8 minutes, perfect overview

**Reading Resource:**
- [RESTful API Design - Best Practices](https://hackernoon.com/restful-api-designing-guidelines-the-best-practices-60e1d954e7c9) - Comprehensive guide with examples

### Real-time Communication Concepts 
- (We will implement this in the next week, you can read this after completing this weeks tasks :))

**Firebase Realtime Database:**
- NoSQL database that syncs data in real-time
- Automatic updates when data changes
- Offline support and data persistence
- No need for a custom server, Firebase acts as a stand-alone full-fledged backend💪

**Resources:**
- [Introducing Firebase Realtime Database](https://firebase.google.com/docs/database)
- [How to Setup Firebase Authentication in MERN Stack App | MERN + FIREBASE](https://www.youtube.com/watch?v=unr4s3jd9qA)

:)

## Week 2 Implementation Tasks

### Task 1: Backend Setup (Node.js + Express)

Set up your Express server foundation that will handle all user management and authentication.

**What You Need to Do:**
- Create Node.js project with Express
- Install essential packages (express, cors, dotenv)
- Set up basic server with middleware
- Create basic route structure

**Minimal Starter:**
```javascript
const express = require('express');
const app = express();
// Add your middleware and routes here
app.listen(5000, () => {
    console.log('Server running on port 5000');
});
```

**Follow These Resources:**
- [Express Hello World](https://expressjs.com/en/starter/hello-world.html)
- [Node.js Project Structure Best Practices](https://blog.logrocket.com/organizing-express-js-project-structure-better-productivity/)

**Essential Packages to Install:**
```bash
npm install express cors dotenv mongoose jsonwebtoken bcryptjs
```

### Task 2: Database Setup (MongoDB)

**Goal:** Connect your Express app to a local MongoDB database

**What You Need to Do:**
- Install MongoDB locally on your computer
- Install mongoose package
- Create database connection
- Test the connection

**Installation Guides:**
- [Install MongoDB on Windows](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-windows/)
- [Install MongoDB on macOS](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/)
- [Install MongoDB on Linux](https://docs.mongodb.com/manual/administration/install-on-linux/)

**Connection Reference:**
- [Mongoose Connect](https://mongoosejs.com/docs/connections.html)

### Task 3: Implementing User Authentication CRUD Operations

**Goal:** Create a complete user management system with RESTful API endpoints

**What You Need to Research:**
- RESTful API design principles and HTTP methods
- Mongoose schema creation and validation
- Password hashing with bcrypt
- Express route handling and error management
- MongoDB query operations

**API Endpoints Design:**
Design and implement RESTful API endpoints for user authentication (/api/auth):
- Define endpoints for Create (Register), Read (Login/Profile), Update, and Delete operations

**Your Tasks:**

**User Model Creation:**
- Create User schema with username, email, password fields
- Add validation rules and constraints
- Include timestamps for tracking

**Authentication Endpoints:**
- POST /api/auth/register - Create new user
- POST /api/auth/login - Authenticate existing user
- GET /api/auth/profile - Get user profile
- PUT /api/auth/profile - Update user profile
- DELETE /api/auth/profile - Delete user account

**Security Implementation:**
- Hash passwords using bcrypt
- Validate input data
- Handle duplicate users
- Secure password comparison

**Key Resources:**
- [RESTful API Design](https://restfulapi.net/)
- [Mongoose Models & Validation](https://mongoosejs.com/docs/validation.html)
- [bcrypt Password Hashing](https://www.npmjs.com/package/bcryptjs)
- [Express Router](https://expressjs.com/en/guide/routing.html)
- [HTTP Status Codes](https://httpstatuses.com/)

**Minimal Route Structure:**
```javascript
// routes/auth.js
const express = require('express');
const router = express.Router();

// POST /api/auth/register
router.post('/register', (req, res) => {
    // Your implementation here
});
// Add other CRUD routes...

module.exports = router;
```

**Success Criteria:**
- User schema created with proper validation
- All CRUD endpoints implemented and working
- Login verifies credentials correctly
- Passwords are hashed before saving
- Proper HTTP status codes returned
- Error handling for invalid requests

### Task 4: Basic Messaging UI/UX (React)

Create the frontend interface for your chat application.

**What You Need to Build:**
- Login/Register forms
- Simple chat interface layout
- Message input component
- User authentication state management
- Basic styling

**Learning Resources:**
- [React Forms Tutorial](https://reactjs.org/docs/forms.html) - Form handling
- [React State Management](https://reactjs.org/docs/hooks-state.html) - useState & useEffect
- [React Authentication Flow](https://blog.logrocket.com/complete-guide-authentication-with-react-router-v6/) - Managing auth state

**Key Components:**
- Login/Register forms
- Chat interface layout
- Message display area
- Message input field
- Navigation between views

## Testing and Troubleshooting

### Verification Checklist

- [ ] Express server running without errors
- [ ] MongoDB connection established successfully
- [ ] User registration endpoint working (`POST /api/auth/register`)
- [ ] User login endpoint working (`POST /api/auth/login`)
- [ ] Password hashing implemented with bcrypt
- [ ] JWT tokens generated on successful login
- [ ] All CRUD operations tested with Postman

### Using Postman for API Testing

Test your Express endpoints:
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/profile` - Get user profile (protected)
- `GET /users` - Get users list

**Tutorial:** [Testing APIs with Postman](https://learning.postman.com/docs/getting-started/sending-the-first-request/)

## Bonus Tasks 

### Bonus 1: Advanced Password Security
- Encrypt Password while saving to the Database
- Add password reset functionality via email

**Tutorial:** [Advanced Password Security](https://auth0.com/blog/hashing-passwords-one-way-road-to-security/)

### Bonus 2: Advanced Authentication
- Implement a refresh token mechanism
- Add email verification for new users

**Tutorial:** [Advanced JWT Implementation](https://auth0.com/blog/a-look-at-the-latest-draft-for-jwt-bcp/)

### Bonus 3: Database Optimization
- Add database indexing
- Implement data validation middleware
- Add database query optimization

## Common Issues & Troubleshooting

**Issue: MongoDB Connection Problems**
- Solution: Check connection string and network access settings
- Resource: [MongoDB Connection Troubleshooting](https://www.mongodb.com/docs/drivers/node/current/connection-troubleshooting/)

**Issue: CORS Errors**
- Solution: Make sure you have CORS middleware configured properly
- Resource: [Understanding CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)

**Issue: JWT Token Issues**
- Solution: Check token expiration and secret key
- Resource: [JWT Debugging Guide](https://jwt.io/introduction)

## Conclusion

Week 2 is where your chat application starts to come alive with professional backend development! Focus on getting the Express + MongoDB authentication system working properly . Don't worry about perfection - the goal is to have a solid foundation with both technologies working together.

**Remember to:**
- Test each component as you build it
- Use the provided resources when you get stuck
- Focus on the main tasks first, save bonus features for later
- Ask for help if you're struggling with concepts
- Document your progress and any issues you encounter.

Next week, we'll make your chat app more robust with advanced features and better security. Keep coding! 🚀
