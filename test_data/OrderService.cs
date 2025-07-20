using System;
using System.Collections.Generic;
using System.Linq;

namespace TestProject 
{
    public class Order
    {
        public int Id { get; set; }
        public User Customer { get; set; }
        public List<OrderItem> Items { get; set; }
        public DateTime OrderDate { get; set; }
        public decimal TotalAmount => Items.Sum(item => item.Price * item.Quantity);
        
        public Order(int id, User customer)
        {
            Id = id;
            Customer = customer;
            Items = new List<OrderItem>();
            OrderDate = DateTime.Now;
        }
        
        public void AddItem(Product product, int quantity)
        {
            var item = new OrderItem
            {
                Product = product,
                Quantity = quantity,
                Price = product.Price
            };
            
            Items.Add(item);
        }
        
        public override string ToString()
        {
            return $"Order: {Id}, Customer: {Customer.Name}, Items: {Items.Count}, Total: {TotalAmount:C}";
        }
    }
    
    public class OrderItem
    {
        public Product Product { get; set; }
        public int Quantity { get; set; }
        public decimal Price { get; set; }
    }
    
    public class OrderService
    {
        private List<Order> _orders;
        
        public OrderService()
        {
            _orders = new List<Order>();
        }
        
        public void AddOrder(Order order)
        {
            _orders.Add(order);
        }
        
        public Order GetOrderById(int id)
        {
            return _orders.FirstOrDefault(o => o.Id == id);
        }
        
        public List<Order> GetOrdersByUser(User user)
        {
            return _orders.Where(o => o.Customer.Id == user.Id).ToList();
        }
    }
}
