#include "model.h"
void Model::Render() {}
void Model::Render(Position3D position){}
VegetationData::VegetationData() {
m_MeshData.assign({5,1,2,8,2,9});
  m_Texture = R"(
    #
   ###
  #####
    #
    #
    #
)";
}
const char* VegetationData::GetTexture() const {
  return m_Texture;
}
const std::vector<int>& VegetationData::GetMeshData() const {
  return m_MeshData;
}
Vegetation::Vegetation(std::string_view tint, 
                       VegetationData *p) 
            : m_Tint{tint}, m_pVegData{p} {
  ++m_Count;
            }
void Vegetation::Render(Position3D position) {
  std::cout << m_pVegData->GetTexture();
  std::cout << "Mesh: " ;
  for (auto m: m_pVegData->GetMeshData()) {
    std::cout << m << " ";    
  }
  std::cout << "\nTint" << m_Tint << std::endl;
  std::cout << "Position: " << position << std::endl;
}
//void Vegetation::Render(Position3D position){}
void Vegetation::ShowCount() {
  std::cout << "Total objects created: " << m_Count << std::endl;
}
VegetationPtr VegetationFactory::GetVegetation(std::string_view tint) {
  auto found = m_Flyweights.find(tint) != end(m_Flyweights);
  if (!found) {
    m_Flyweights[tint] = std::make_shared<Vegetation>(tint, m_pVegData);
  }
  return m_Flyweights[tint];
}
int main() {
  std::vector<VegetationPtr> m_Trees{};
  VegetationData data{};
  VegetationFactory factory{&data};
  for(int i=0;i<15;++i) {
    if (i<5) {
      m_Trees.push_back(factory.GetVegetation("Green"));
      m_Trees[i]->Render({i*10,i*10,i*10});
    } else if (i>5 && i <= 10) {
      m_Trees.push_back(factory.GetVegetation("Dark Green"));
      m_Trees[i]->Render({i*10,i*10+10,i*10});
    } else {
      m_Trees.push_back(factory.GetVegetation("Light Green"));
      m_Trees[i]->Render({i*10+10,i*10,i*10});
    }
  }
  Vegetation::ShowCount();
  return 0;
}