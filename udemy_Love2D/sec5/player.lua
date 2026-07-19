playerStartX = 360
playerStartY = 100
player = world:newRectangleCollider(playerStartX,playerStartY,40,100, {collision_class = "Player"})
player:setFixedRotation(true) -- avoid the object rotation when it falls from the edge of a platform
player.speed = 240 -- max at 60FPS
player.animation = animations.idle
player.isMoving = false
player.direction = 1
player.grounded = true
  
function playerUpdate(dt)
  if player.body then
    local colliders = world:queryRectangleArea(player:getX()-20, player:getY()+50, 40, 2, {'Platform'})    
    if #colliders > 0 then
      player.grounded = true
    else
      player.grounded = false
    end
    player.isMoving = false -- default value
    local px,py = player:getPosition()
    if love.keyboard.isDown('right') then
      player:setX(px + player.speed*dt)
      player.isMoving = true
      player.direction = 1
    end
    if love.keyboard.isDown('left') then
      player:setX(px - player.speed*dt)
      player.isMoving = true
      player.direction = -1
    end
    if player:enter('Danger') then
      player:setPosition(playerStartX, playerStartY)
    end
  end
  if player.grounded then
    if player.isMoving then
      player.animation = animations.run
    else
      player.animation = animations.idle
    end
  else
    player.animation = animations.jump
  end
  player.animation:update(dt)
end 
function drawPlayer()
  local px,py = player:getPosition()
  player.animation:draw(sprites.playerSheet,px,py, nil, 0.25*player.direction, 0.25, 130,300)
end