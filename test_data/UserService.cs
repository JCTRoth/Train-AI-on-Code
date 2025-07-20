using System;
using System.Collections.Generic;
using System.Linq;

namespace TestProject 
{
    public class User
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Email { get; set; }
        
        public User(int id, string name, string email)
        {
            Id = id;
            Name = name;
            Email = email;
        }
        
        public override string ToString()
        {
            return $"User: {Id}, {Name}, {Email}";
        }
    }
    
    public class UserService
    {
        private List<User> _users;
        
        public UserService()
        {
            _users = new List<User>();
        }
        
        public void AddUser(User user)
        {
            _users.Add(user);
        }
        
        public User GetUserById(int id)
        {
            return _users.FirstOrDefault(u => u.Id == id);
        }
        
        public List<User> GetAllUsers()
        {
            return _users.ToList();
        }
    }
}
