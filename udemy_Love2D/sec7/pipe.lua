pipes = {}

-- Spawns a pair of pipes (top and bottom)
function spawnPipes()

    math.randomseed(os.time())
    local spawnRef = math.random(200, 700) + shiftDown
    spawnPipe(spawnRef, true)
    spawnPipe(spawnRef, false)

end

-- Spawns a single pipe, using 'top' to determine its position
function spawnPipe(spawnRef, top)

    local pipeX = coreWidth + 100
    local pipeY = spawnRef
    if top then
        pipeY = pipeY - 1160
    else
        pipeY = pipeY + 160
    end

    local pipeCollider = world:newRectangleCollider(pipeX, pipeY, 104, 1000, {collision_class = "Pipe"})
    pipeCollider:setType('static')
    pipeCollider.top = top
    pipeCollider.passed = false
    pipeCollider.delete = false
    table.insert(pipes, pipeCollider)

end

function updatePipes(dt)

    for i,p in ipairs(pipes) do
        p:setX(p:getX() - (140 * dt))

        -- passsed tells us if the bird has passed a pipe or not
        if p:getX() < birdCollider:getX() and p.passed == false and p.top then
            score = score + 1
            p.passed = true
        end

        -- delete the pipe if it goes offscreen (left)
        if p:getX() < -100 then
            p.delete = true
        end
    end

    destroyDeletedPipes()

end

function drawPipes()

    for i,p in ipairs(pipes) do
        local scaleY = 1
        if p.top then
            scaleY = -1
        end
        love.graphics.draw(sprites.pipe, p:getX(), p:getY(), nil, nil, scaleY, sprites.pipe:getWidth()/2, sprites.pipe:getHeight()/2)
    end

end

function destroyDeletedPipes(deleteAll)

    for i,p in lume.ripairs(pipes) do
        if p.delete or deleteAll then
            p:destroy()
            table.remove(pipes, i)
        end
    end

end
