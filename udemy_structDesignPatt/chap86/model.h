#pragma once
#include <iostream>
#include <string_view>
#include <vector>
#include <memory>
#include <unordered_map>
struct Position3D {
  int x,y,z;
  friend std::ostream& operator<< (std::ostream&os, const Position3D& obj) {
    return os << "{" << obj.x <<","<<obj.y <<","<<obj.z <<")\n";
  }
};
class Model {
public:
  virtual void Render();
  virtual void Render(Position3D position);
};
class VegetationData {
  std::vector<int> m_MeshData{};
  const char *m_Texture{};
public:
  VegetationData();
  const char *GetTexture() const;
  const std::vector<int> & GetMeshData() const;
};  
class Vegetation: public Model {
  inline static int m_Count {};
  VegetationData *m_pVegData{};
  std::string m_Tint{};
public:
  Vegetation(std::string_view tint, VegetationData *p);
  void Render(Position3D position) override;
  static void ShowCount();
};
using VegetationPtr = std::shared_ptr<Vegetation>;
class VegetationFactory {
  std::unordered_map<std::string_view, VegetationPtr> m_Flyweights{};
  VegetationData * m_pVegData{};
public:
  VegetationFactory(VegetationData* mPVegData) : m_pVegData{mPVegData}{}
  VegetationPtr GetVegetation(std::string_view tint);
};