## Master the Art of Managing Projects with Azure DevOps
- Instructor: Mark Lee

## Section 1: Azure DevOps Essentials: Getting Started

### 1. Lesson 1: Why Learn Azure DevOps!

### 2. Lesson 2: A Course Overview
- E-book
  - An Azure DevOps Playbook for Project management

### 3. Lesson 3: A Brief History of Microsoft Azure DevOps
- Devops: delivering value by streamlining the SW development and delivery process
- History
  - Team Foundation Server (TFS) at 2005
  - Team Foundation Service at cloud at 2011
  - Azure DevOps at 2018
- Azure boards: dash board
- Azure Repositories: Git
- Azure Pipelines
- Azure Test Plans  
- Azure Artifacts

### 4. Lesson 4: Let's Create a Free MS Azure DevOps Account
- Using github account was looping forever
- Use a regular main account to create a new account
- After the account creation, search Azure DevOps
  - Creat a new project of Customer Chat Bot

### 5. Lesson 5: What is an Organization in Azure DevOps
- Click Azure DevOps icon in Top-Left then Organization Settings in Bottom-Left
  - Can adjust Time Zone
  - One organization for a personal account

### 6. Lesson 6: How a Project is related to an Azure DevOps Organization
- When creating a project, click advanced menu and configure Process as Basic or Agile as necessary
  - Changing Process later in the existing project is not recommended as manual adjustment is required
  - Make a new process inheriting the current one and the replacement would be OK
  - To change the current process, Organization settings -> All processes -> Select the current one -> Projects -> Click (...) of the corresponding project to change the process

### 7. Lesson 7: What are the Azure DevOps User Access Levels


### 8. Lesson 8: An Overview of the Group Permission's in Azure DevOps
- Create a PMO group
  - Project related group is not visible in organization level

### 9. Lesson 9: Customizing Your Azure DevOps Profile
- User setting in Top-right icon

### 10. Lesson 10: An Overview of the Hierarchical Process Taxonomy
- Epic: a larger body of work to be broken down into smaller, manageable tasks or user stories to facilitate planning and trackign within a project
- Feature: a high level requirement that provides business value and can be broken down into user stories
- User story: a high-level requirement based on the user experience that provides business value and is broken down into tasks
- Task: a specific unit of work that represents an actionable item to be completed within a project, often associated with a user story
- Ref: https://learn.microsoft.com/en-us/azure/devops/boards/work-items/guidance/choose-process?view=azure-devops&tabs=agile-process
- Basic:

<img src="./basic-process-epics-issues-tasks-2.png" height="200">

- Agile:

<img src="./alm_pt_agile_wit_artifacts.png" height="300">

- SCRUM:

<img src="./alm_pt_scrum_wit_artifacts.png" height="300">

- CMMI:

<img src="./alm_pt_cmmi_wit_artifacts.png" height="300">


## Section 2: Starting with the Basics: Creating ADO Work Items

### 11. Lesson 11: An Introduction to Section 2

### 12. Lesson 12: Navigating and Understanding the Azure Board
- KANBAN vs SCRUM: Both for agile
- Kanban
  - No deadlines!
  - A continuous flow
  - Ongoing or rolling basis work
  - Best suited for ITsupport teams and operations
- Scrum
  - Time-boxed (1 to 4 weeks)
  - More rigid approach
  - Best suited for product development or projects with evolving requirements
- Work in Progress (WIP)
  - The number of unfinished tasks, user stories, or features that a team has started but not yet completed
  - WIP limit: a cap on how many items can be in a specific workflow

### 13. Lesson 13: Creating Epics and Reviewing Work Item Properties
- In Boards, only Features/Stories are visible by default
  - To see Epic, Project settings -> Boards -> Team configuration, opt in **Epics**

<img src="./ch13_board.png" height="150">

- For each board item, edit Stats, Reasons, Area, Iteration, Planning, ... as necessary
  - Once the state is Removed, it cannot be back to intermediate states

### 14. Lesson 14: Creating Features and Introducing User Stories
- In Boards, enable Epic as visible
  - In Epic item, you can add features
- Essential Scrum by Kennath Rubin
  - Story must be INVEST
    - I for Independent
    - N for Negotiable
    - V for Valuable
    - E for Estimated
    - S for Sized Appropriately
    - T for Testable

### 15. Lesson 15: The Anatomy of a User Story-Understanding Its Structure
- A user story describes a feature of functionality from an end-user's perspective
- In contrast to a user stody, a requirement is a statement that specifies what a system or product must do or how it must behave to meet specific criteria
- A sample user story title: "As a customer service representative, I want a chatbot implemented for common industries, so that the process can be streamlined and reduce the load on Customer service representative"
- Acceptance Criteria

### 16. Lesson 16: Creating the User Story

### 17. Lesson 17: How to Create a Task, Test Case, and Bug Work Items

### 18. Lesson 18: Using AI to Generate Work Items
- See the attached excel file

### 19. Lesson 19: Creating the Iteration or Sprint with an Overview of the Task Board
- Project settings -> Project configuration -> New Iteration -> Add Sprint 4

## Section 3: Achieving Consistency: Customizing Project Processes, Work Items, and Backlogs

### 20. Lesson 20: An Introduction to Section 3

### 21. Lesson 21: The Benefits of Customizing Azure DevOps Projects: A Game-Changer
- Alignment with organizational process
- Tailored workflows
- Granular control
- Improved visibility and reporting

### 22. Lesson 22: Customizing the ADO Project Process
- At Organization settings
  - Changing the organization name 
  - Can create inherited process from Process menu
  - Adding new states in User story
- ADO crashed with Opera
  
### 23. Lesson 23: Mastering Azure Board States Customization
- Adding new states into User story from Organization Settings->Process
- Adding columns in Boards using settings in Boards -> Boards in each Project

### 24. Lesson 24: How to Add Values to Work Item Pick Lists in Azure DevOps

### 25. Lesson 25: Enhancing Azure DevOps Work Items with Custom Fields

### 26. Lesson 26: Creating a Work Item Rule
- Organization settings->Process->All processes->ADOAgileProcess->Task->Rules

### 27. Lesson 27: Adding a New Page to a Custom Work Item

### 28. Lesson 28: Unlocking Flexibility: Creating Custom Work Items in Azure DevOps

### 29. Lesson 29: Creating Color-Coded Tags
- Project settings -> Boards -> User Story - Add Tags
- Tag colors can be changed from Boards -> Settings -> Tag colors

### 30. Lesson 30: Mastering Project Wikis in Azure DevOps
- Overview -> Wiki
- Markdown format
- Using "Mention a work item", each Epic/Feature/User story can be injected

### 31. Lesson 31: Using the Copy Work Item Feature

### 32. Lesson 32: How to Standardize Your Process with Templates in Azure DevOps
- Project Settgings -> Team configuration -> Templates
  - For every work item like Bug, Epic, Feature, User Story, ...

### 33. Lesson 33: Exploring More Customization Options for Azure Boards

## Section 4: Getting Started with Azure DevOps Queries

### 34. Lesson 34: An Introduction to Section 4

### 35. Lesson 35: ADO Queries Part 1 (What are ADO Queries)
- ADO Queries
  - Flat
  - Tree
  - Linked
- Benefits of using queries
  - Prioritization and planning
  - Cross-Project queries
  - Query tags
  - Using query reports (aggregated work item data)  
  - Tracking dependencies
  - Quality assurance (severity, test status, code review status)
  - A primary purpose of queries is for dashboarding
- Project -> Boards -> Queries

### 36. Lesson 36: ADO Queries Part 2: Creating a Flat Query

### 37. Lesson 37: ADO Queries Part 3: Flat Queries Continued and Linked Queries
- Changing query types:

<img src="./ch38_queries.png" height="200">

### 38. Lesson 38: ADO Queries Part 4: Exploring the Tree Query

### 39. Lesson 39: ADO Queries Part 5: Visualize Charts, Format Reports, Bulk Editing
- Boards -> Queries -> My Queries -> Results
  - Preview option is available
- After queries, selecting items then we can change their column values altogether

## Section 5: Step-by-Step Process for Building a Project from the Ground Up!

### 40. Lesson 40: An Introduction to Section 5
- Installing extension for visualization

### 41. Lesson 41: Building an ADO Project Together: Hands-On Tutorial
- Board columns as New, Analysis, Design, Development, In Testing, Blocked
- Then using the enclosed excel sheet, add Epic/Features/User story items
  - In Boards -> Backlogs

<img src="./ch41_items.png" height="300">

### 42. Lesson 42: Estimating the Scope of Work in Our Project
- Each task
  - Original estimate as 10 hours

### 43. Lesson 43: Setting Up Our Backlog to View Task Estimates
- 30 hour for each story points

### 44. Lesson 45: Let's Install and Explore the 'Visualize' Extension
- Organization settings
  - Market place in Top-right menu
  - Select "Work Item Visualization"
- Now Boards -> Backlogs -> Select Epic item -> Top-right More Actions (:) -> Visualize

<img src="./ch44_tree.png" height="500">

### 45. Lesson 46: Final Thoughts on Assigning Work Across Sprints

### 46. Lesson 47: Setting Up the Sprint Resource Capacity Model
- Test or dummy users can be added from Organization settings
  - Email needs not to be valid
- Allocate hours in Boards -> Sprints -> Capacity in each person  

### 47. Lesson 44: How to Set Sprint Dates and Plan Work for Each Sprint

### 48. Lesson 48: Why You Should Standardize the Backlog for Better Project Management

### 49. Lesson 49: How to Create a Strategic-Level Work Item in the Backlog
- Organization settings -> Process -> Backlog levels
  - Adding Initiative as a higher hierarchy
- An existing Epic can be located below this higher item using Add Link menu in the Epic item setting  

### 50. Lesson 50: Final Thoughts on Customizing the Backlog for Maximum Impact
- Organization settings -> Board -> Processes -> Backlog levels
  - Default Agile/Basic/Scrum do not allow changes
  - Make a new process inheriting one of them. Then backlog levels/work items can be edited

### 51. Lesson 51: Creating a Delivery Plan: Tracking Dependencies Along the Way
- Dependency track from Market Place
- Visualizes work across teams and ensures timelines align with goals

### Assignment 1: Building an ADO Project from the Ground Up!
1. Create a Project and assign the Agile process. This step is unnecessary if you've already created a default project when setting up your ADO Organization.
2. Create a custom child process based on the Agile parent process.
3. Install the "Visualize" extension from the Azure DevOps Marketplace.
4. Create Work Items for your project using the provided MS Excel resource called "Customer Service AI Chatbot Project Work Items" file.
5. Verify your Backlog to ensure it matches the items in the worksheet.
6. Set iteration dates or rename them as Sprints in the Project Configuration section of ADO.
7. Add Iterations to the project through the Team Configuration section in ADO.
8. Enable the Planning Pane in the Backlog view by selecting "View Options" and toggling it on, so the Iterations are visible.
9. Assign planned work to your Iterations as I've demonstrated how to do.
10. Create a custom work item named "Compliant" and add a field called "Compliance" with the following values: HIPP, SOX, GXP. Then, add this field to the Details page.
11. Next, create a new "Compliant" work item to verify that the custom work item and its associated field and values were successfully created.
12. Link the custom "Compliant" work item to any User Story.
13. Lastly, go to the Organization Settings menu and add the "Compliant" work item to the Iteration Backlog located under `Boards/Process/<select your custom process>`, then navigate to the tab called "Backlog levels.
14. Confirm that the "Compliant" work item is linked to the User Story is displayed in the Backlog as part of the hierarchy.

### 52. Lesson 52: Navigating ADO Analytic Views for Better Insights

## Section 6: Mastering ADO Dashboards and Queries: A Deep Dive

### 53. Lesson 53: An Introduction to Section 6

### 54. Lesson 54: How to Navigate ADO Dashboards: An Introduction
- Widget catalogue
- Dashboard can be visible to other teams or team members

### 55. Lesson 55: Building Dashboards with Corresponding Queries: Part 1
- Work Item query
  - Shows a list of work items based on a predefined query

### 56. Lesson 56: Building Dashboards with Corresponding Queries: Part 2
- Widget settings can refresh dashboard every 5min

### 57. Lesson 57: Building Dashboards with Corresponding Queries: Part 3

### 58. Lesson 58: Building Dashboards with Corresponding Queries: Part 4

### 59. Lesson 59: Building Dashboards with Corresponding Queries: Part 5

### 60. Lesson 60: Building Dashboards with Corresponding Queries: Part 6
- Velocity
- Burndown

### 61. Lesson 61: Building Dashboards with Corresponding Queries: Part 7

### 62. Lesson 62: Building Dashboards with Corresponding Queries: Part 8

### 63. Lesson 63: Building Dashboards with Corresponding Queries: Part 9

### 64. Lesson 64: The Process of Exporting and Importing Work Items in ADO

## Section 7: Azure Test Plans

### 65. Lesson 65: The Basics of Testing with Azure DevOps
- Functional testing
  - Unit testing
  - Iterative testing
    - Across spreads
    - Early detection of issues
    - Increased Quality
  - Integration or End-to-End Testing
    - Comprehensive validation of requirements
    - Validation of non-functional requirements
  - System Testing
  - Regression Testing
  - User acceptance Testing (UAT)
  - Smoke Testing
- Non functional testing
  - Performance Testing
  - Security Testing
  - Usability Testing
  - Compatibility Testing

### 66. Lesson 66: Installing the Test and Feedback Extension
- Test Plans in each Project
- Test & Feedback from Market place
  - Supports Chrome and Edge browser
  - Supporting Firefox will be retired Nov 2026
- Not free?
  - 30 days free trial only

### 67. Lesson 67: Install Azure Test Plans and Create our first Test Plan
- Azure Test Plans is a management tool but not a testing tool. Testing must be coupled with Azure Pipelines

### 68. Lesson 68: Explore Executing Test Cases with the Web App called Test Runner

### 69. Lesson 69: Let's Create a Test Case Shared Step(s) for Repeatability

### 70. Lesson 70: Let's Create a Test Case Shared Parameter for Repeatability

### 71. Lesson 71: Creating and Assigning Test Configurations for Repeatability

### 72. Lesson 72: Building a Requirements-Based Test Suite in Azure Test Plans

### 73. Lesson 73: Creating a Static and Query Based Test Suite

### 74. Lesson 74: Creating Test Management Charts and add to a Dashboard

### 75. Lesson 75: We'll Conduct Exploratory Testing with the Test & Feedback Extension

## Section 8: A Beginner's Guide to Continuous Integration and Delivery

### 77. Lesson 76: An Introduction to Section 8

### 78. Lesson 77: Starting Your Journey with Azure DevOps CI and CD
- Ref: https://learn.microsoft.com/en-us/azure/devops/?view=azure-devops
- The anatomies of Azure DevOps CI/CD (automation)
  - Azure Repo with Git & GitHub
  - A Deep Dive into Branching
  - The anatomy of the build pipeline
  - The anatomy of a YAML for automation
  - The anatomy of the release pipeline
  - Connecting the dots an end-2-end build & release scenario
- Continuous Integration
  - Developers write and commit code to a repo frequently
  - Automated build process compiles the code and run tests to create Artifacts
  - Automated tests (unit-tests, integration tests) to validate the code changes
  - Package the compiled code and dependencies in deployable packages
- Continuous Delivery
  - Automated deployment processes push the build Artifacts to straging or production environment
  - Automating the Release of the SW to suers or customers
  - Monitoring the Application in production to ensure it functions correctly

### 79. Lesson 78: Getting to Know Git: Distributed Code Management Demystified
- Git: a version control system that allows for distributed code management
- TFVC is still running
- Git anatomy
  - Commits: a snapshot of changes made to files in a repo
  - Fetch: retrieves updates from a remote repo without merging them into the local branch
  - Fork: a personal copy of a repo
  - Clone: creates a local copy of a remote repo
  - Push: uploads local commints from your repo to a remote repo
  - Pull-Request: merge changes from one branch into another
  - Merge: a process that combines changes from different branches into a single branch
  - Branching: enables multiple features or fixes to be developed simultaneously without interfering with the main codebase

### 80. Lesson 79: Creating Your First "Git" Repo in Azure DevOps
- Create a new project
- Create a new repo from Repos -> Files -> New Repositories
  - Default README.md is created

### 81. Lesson 80: How to Create Your First Branch in Azure DevOps


### 82. Lesson 81: Creating Your First Pull Request in Azure DevOps


### 83. Lesson 82: How to install the Microsoft IDE (Integrated Development Environment)


### 84. Lesson 83: Creating a Web Project with Git Repository in Your IDE


### 85. Lesson 84: How to Clone and Fork a Repo: Understanding the Process


### 86. Lesson 85: Getting Started with Build Pipelines: Understanding the YAML File


### 87. Lesson 86: How to Set Up Your First ADO Pipeline for a Web Application


### 88. Lesson 87: How to Publish Your Web App Build Artifact with YAML


### 89. Lesson 88: How to Set Up a Release Pipeline with Azure Web App Service


### 90. Lesson 89: Automating Your Pipeline Workflow from Start to Finish


### 91. Lesson 90: How to Build a Multi-Stage Release Pipeline


### 92. Lesson 91: Creating a Parallel Release Pipeline and Setting Up Release Approvals


### 93. Lesson 92: How to Create Pipeline Deployment Gates for Conditional Deployment


### 94. Lesson 93: Building a Simple Dashboard for Build and Release History


### 95. Lesson 94: Exploring the Azure DevOps Service: Artifacts


### 96. Lesson 95: Closing Thoughts on Continuous Integration and Delivery


Not completed
Start
Quiz 8: Assess Your Understanding of Continuous Integration and Delivery
Not completed
Start
97. Exercise: CI/CD Pipeline Hands-On Practice - Optional (Downloadable Resource)



### 98. Lesson 96: An Introduction to Section 9


### 99. Lesson 97: Part 1: Using Artificial Intelligence to Automate Work Item Creation


### 100. Lesson 98: Part 2: Configuring the AI Work Item Assistant for Automation


### 101. Lesson 99: Part 1: GitHub Integration with Azure DevOps


### 102. Lesson 100: Part 2: GitHub Integration with Azure DevOps


### 103. Lesson 101: Part 3: GitHub Integration with Azure DevOps


### 104. Lesson 102: Part 1: Setting Up Msft Teams Integration with Azure DevOps Boards


### 105. Lesson 103: Part 2: Setting Up Msft Teams Integration with Azure DevOps Boards


### 106. Lesson 104: Part 1: Connecting MS Excel with Azure DevOps


### 107. Lesson 105: Part 2: Connecting MS Excel with Azure DevOps


Not completed
Start
Quiz 9: Evaluate Your Knowledge on Azure DevOps Integrations

### 108. Lesson 106: An Introduction to Section 10


### 109. Lesson 107: Part 1: Setting Up Scaled Agile in Azure DevOps


### 110. Lesson 108: Part 2: Setting Up Scaled Agile in Azure DevOps


### 111. Lesson 109: Part 3: Setting Up Scaled Agile in Azure DevOps


### 112. Lesson 110: Part 4: Setting Up Scaled Agile in Azure DevOps


### 113. Lesson 111: Part 5: Setting Up Scaled Agile in Azure DevOps


Not completed
Start
Quiz 10: Test Your Knowledge on Setting Up Scaled Agile in Azure DevOps

### 114. Lesson 112: An Introduction to Section 11


### 115. Lesson 113: Explore the PMI Project Management Model in Azure DevOps


### 116. Lesson 114: Understanding the Agile Manifesto in the Context of Azure DevOps


### 117. Lesson 115: How the Stages of Team Formation Apply in Azure DevOps


### 118. Lesson 116: Scrum Ceremonies and Their Role in Azure DevOps


### 119. Lesson 117: Understanding the Definition of Done (DOD) with an Extension


### 120. Lesson 118: Playing Planning Poker in Azure DevOps to Estimate Work


### 121. Lesson 119: The Retrospective Extension: A Tool for Continuous Improvement


### 122. Lesson 120: Part 1: Managing Timesheets in Azure DevOps with an Extension


### 123. Lesson 121: Part 2: Managing Timesheets in Azure DevOps with an Extension


### 124. Lesson 122: Something Extra: Future Proofing Your Career in an AI Dr


Not completed
Start
Quiz 11: Test Your Knowledge of Various Azure DevOps Extensions
### 125. Bonus Section


Not completed
Start
126. Claiming PMI PDU's for this Course


