#pragma once
#include <string>
#include <memory>
#include <experimental/filesystem>
class Employee {
  std::string m_Name;
  std::string m_Role;
public:
  Employee(const std::string& name, const std::string &role);
  const std::string& GetName() const;
  const std::string& GetRole() const;
  std::string GetInfo() const;
};
class Storage {
public:
  virtual void CreateFile(const std::string &fileName) = 0;
  virtual void DeleteFile(const std::string &fileName) = 0;
  virtual void UpdateFile(const std::string &fileName) = 0;
  virtual void ViewFile(const std::string &fileName) = 0;
  virtual ~Storage() = default;
};
class Repository : public Storage {
  std::shared_ptr<Employee> m_pEmp;
  std::experimental::filesystem::path m_CurrentPath{};
public:
  std::shared_ptr<Employee> GetUser() const {
    return m_pEmp;
  }
  void SetEmployee(std::shared_ptr<Employee> p) {
    m_pEmp = p;
  }
  Repository(const std::string &repoPath);
  void CreateFile(const std::string& fileName) override;
  void DeleteFile(const std::string& fileName) override;
  void UpdateFile(const std::string& fileName) override;
  void ViewFile(const std::string& fileName) override;
};
