function love.load()
  -- initializer
  target = {}
  target.x = 300
  target.y = 300
  target.radius = 50
  score = 0
  timer = 0
  gameState = 1 -- 1 for main menu, not running game. 2 for game mode
  --
  gameFont = love.graphics.newFont(40)
  sprites = {}
  sprites.sky = love.graphics.newImage('sprites/sky.png')
  sprites.target = love.graphics.newImage('sprites/target.png')
  sprites.crosshairs = love.graphics.newImage('sprites/crosshairs.png')
  love.mouse.setVisible(false) -- hide mouse point
end
function love.update(dt) -- dt: delta time
  -- 60 FPS might not be guaranteed
  -- dt changes by varying FPS
  if timer > 0 then
    timer = timer - dt
  end
  if timer < 0 then
    timer = 0
    gameState = 1
  end
end
function love.draw()
  --
  love.graphics.draw(sprites.sky,0,0)
  love.graphics.setColor(1,1,1)
  love.graphics.setFont(gameFont)
  love.graphics.print("Score: " .. score,5,5)
  love.graphics.print("Time: " .. math.ceil(timer),300,5)
  if gameState == 1 then
    love.graphics.printf("Click anywhere to begin!", 0,250,love.graphics.getWidth(),"center")
  end
  if gameState == 2 then
    love.graphics.draw(sprites.target, target.x-target.radius,target.y-target.radius)
  end
  love.graphics.draw(sprites.crosshairs,love.mouse.getX()-20,love.mouse.getY()-20) 
end
function love.mousepressed(x,y,button,istouch,presses)
  --
  if button == 1 and gameState == 2 then 
    local mouseToTarget = distanceBetween(x,y,target.x,target.y)
    if mouseToTarget < target.radius then
      score = score + 1
      target.x = math.random(target.radius,love.graphics.getWidth()-target.radius)
      target.y = math.random(target.radius,love.graphics.getHeight()-target.radius)
    end
  elseif button == 1 and gameState == 1 then
    -- Game start or reset
    gameState = 2
    timer = 10
    score = 0
  end
end
function distanceBetween(x1,y1,x2,y2)
  return math.sqrt( (x2-x1)^2 + (y2-y1)^2 )
end