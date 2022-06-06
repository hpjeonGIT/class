#pragma once
#include <string>
#include <iostream>
#include <random>
class Animation {
  std::string m_AnimationData{} ;
public:
  Animation()=default;
  Animation(std::string_view animFile);
  const std::string & GetAnimationData() const {
    return m_AnimationData;    
  }
  void SetAnimationData(const std::string &animationData)  {
    m_AnimationData = animationData;
  }
};
struct Position {
  int x;
  int y;
  friend std::ostream & operator<< (std::ostream &out, Position p) {
    return out << "(" << p.x << ',' << p.y << ')';
  }
};
class Vehicle {
  int m_Speed{};
  int m_HitPoints{};
  std::string m_Name{};
  Animation *m_pAnimation{};
  Position m_Position{};
public:
  Vehicle(int mSpeed, int mHitPoints, const std::string& mName, 
          std::string_view animFile, const Position& mPosition);
  virtual ~Vehicle();
  Vehicle(const Vehicle &other); // copy constructor
  Vehicle &operator=(const Vehicle &other);  // assign operator
  Vehicle(Vehicle &&other) noexcept; // Move constructor
  Vehicle & operator=(Vehicle &&other) noexcept; // move operator
  int GetSpeed() const {    return m_Speed;  }
  int GetHitPoints() const { return m_HitPoints; }
  const std::string& GetName() const {    return m_Name;  }
  Position GetPosition() const { return m_Position; }
  const std::string & GetAnimation() const;
  void SetSpeed(int speed) {m_Speed = speed; }
  void SetPosition(Position position) { m_Position = position; }
  void SetName(const std::string &name) { m_Name = name; }
  void SetHitPoints(int hitPoints) { m_HitPoints = hitPoints; }
  void SetAnimationData(const std::string &animData);
  virtual void Update() = 0;
  virtual Vehicle * Clone() = 0;
};
class GreenCar : public Vehicle {
  using Vehicle::Vehicle;
public:
  void Update() override;
  Vehicle * Clone() override;
};
class RedCar : public Vehicle {
  using Vehicle::Vehicle;
  float m_SpeedFactor{1.5f};
  std::default_random_engine m_Engine{100};
  std::bernoulli_distribution m_Dist{.5};
public:
  void SetSpeedFactory(float factor) { m_SpeedFactor = factor;}
  void Update() override;
  Vehicle * Clone() override;
};
class BlueBus : public Vehicle {
  using Vehicle::Vehicle;
  std::default_random_engine m_Engine{500};
  std::bernoulli_distribution m_Dist{0.5};
public:
  void Update() override;
  Vehicle * Clone() override;
};
class YellowBus : public Vehicle {
  using Vehicle::Vehicle;
  std::default_random_engine m_Engine{500};
  std::bernoulli_distribution m_Dist{0.5};
public:
  void Update() override;
  Vehicle * Clone() override;
};
class GameManager{
  std::vector<Vehicle*> m_Vehicles{};
public:
  void Run();
  ~GameManager();
};
Vehicle * Create(std::string_view type, int mSpeed, int mHitPoints, const std::string& mName,
          std::string_view animFile, const Position& mPosition)  {
  if (type == "redcar") {
    return new RedCar{mSpeed, mHitPoints, mName, animFile, mPosition};
  } else if (type == "greencar") {
    return new GreenCar{mSpeed, mHitPoints, mName, animFile, mPosition};
  } else if (type == "yellowbus") {
    return new YellowBus{mSpeed, mHitPoints, mName, animFile, mPosition};
  } else if (type == "bluebus") {
    return new BlueBus{mSpeed, mHitPoints, mName, animFile, mPosition};
  } 
  return nullptr;
}
  