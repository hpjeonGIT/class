function love.load()

	-- Setting the scale and adjusting screen size
    coreWidth = 720
    coreHeight = 960
    scale = 1
    shiftDown = 0
    osString = love.system.getOS()
    if osString == "Android" or osString == "iOS" then
        scale = love.graphics.getWidth()/coreWidth
        shiftDown = (love.graphics.getHeight() - (coreHeight * scale)) / 2 / scale
    else
        scale = 0.6
    end

    love.window.setMode(coreWidth * scale, coreHeight * scale)

    -- Import libraries
    wf = require "libraries/windfield/windfield"
    timer = require "libraries/hump/timer"
    lume = require "libraries/lume/lume"
    require "libraries/show"
    require 'pipe'

    -- Creating the physics world using windfield (wf)
    world = wf.newWorld(0, 1200, false)
    world:addCollisionClass('Bird')
    world:addCollisionClass('Pipe')

    birdCollider = world:newBSGRectangleCollider(coreWidth/2, -100, 40, 40, 10, {collision_class = "Bird"})

    -- All sprites in the game
    sprites = {}
    sprites.bird = love.graphics.newImage('sprites/bird.png')
    sprites.pipe = love.graphics.newImage('sprites/pipe.png')
    sprites.logo = love.graphics.newImage('sprites/logo.png')
    sprites.bg = love.graphics.newImage('sprites/bg.png')

    -- All fonts in the game
    fonts = {}
    fonts.menuFont = love.graphics.newFont("fonts/BalooTamma2-Regular.ttf", 62)

    -- All sound effects in the game
    sounds = {}
    sounds.flap = love.audio.newSource('sounds/flap.wav', "static")
    sounds.hurt = love.audio.newSource('sounds/hurt.wav', "static")
    sounds.flap:setPitch(1.5)

    spawnPipes()
    timer.every(3, spawnPipes)

    -- 0 = main menu, 1 = gameplay
    gamestate = 0
    score = 0

    -- saveData contains all data that will be saved via filesystem
    saveData = {}
    saveData.highScore = 0

    if love.filesystem.getInfo("data.lua") then
        local data = love.filesystem.load("data.lua")
        data()
    end

end

function love.update(dt)

    if gamestate == 1 then
        world:update(dt)
        timer.update(dt)
        updatePipes(dt)

        local birdY = birdCollider:getY()

        -- Game ends if the bird hits a pipe, or flies too high/low
        if birdCollider:enter('Pipe') or birdY < -100 or birdY > coreHeight + shiftDown*2 then
            gamestate = 0
            sounds.hurt:play()

            if score > saveData.highScore then
                saveData.highScore = score
                love.filesystem.write("data.lua", table.show(saveData, "saveData"))
            end
        end
    end

end

function love.draw()

	-- Scaling all graphics
    love.graphics.scale(scale)
    love.graphics.draw(sprites.bg, -10, -10)
    
    -- Uncomment to visually see the collider boxes
    -- world:draw()

    local bx, by = birdCollider:getPosition()
    love.graphics.draw(sprites.bird, bx, by, nil, nil, nil, sprites.bird:getWidth()/2, sprites.bird:getHeight()/2)

    drawPipes()

    love.graphics.setFont(fonts.menuFont)
    love.graphics.printf("Score: " .. score, 0, coreHeight/1.3 + shiftDown, coreWidth, "center")

    if gamestate == 0 then
        drawMenu()
    end

end

function love.mousepressed(x, y)

	-- Transition from the menu to the main game
    if gamestate == 0 then
        gamestate = 1
        score = 0
        destroyDeletedPipes(true)
        birdCollider:setPosition(coreWidth/2, coreHeight/2)
    end

    -- Make the bird flap when the screen is tapped
    birdCollider:setLinearVelocity(0, 0)
    birdCollider:applyLinearImpulse(0, -2000)
    sounds.flap:play()

end

function drawMenu()

    love.graphics.setColor(1,1,1,1)
    love.graphics.draw(sprites.logo, coreWidth/2, coreHeight/3 + shiftDown, nil, nil, nil, sprites.logo:getWidth()/2, sprites.logo:getHeight()/2)

    love.graphics.setFont(fonts.menuFont)
    love.graphics.printf("Tap anywhere to start!", 0, coreHeight/1.5 + shiftDown, coreWidth, "center")

    love.graphics.printf("High Score: " .. saveData.highScore, 0, coreHeight/1.15 + shiftDown, coreWidth, "center")

end
