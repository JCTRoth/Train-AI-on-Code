using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TestProject 
{
    public class Product
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public decimal Price { get; set; }
        public string Description { get; set; }
        
        public Product(int id, string name, decimal price, string description)
        {
            Id = id;
            Name = name;
            Price = price;
            Description = description;
        }
        
        public override string ToString()
        {
            return $"Product: {Id}, {Name}, {Price:C}";
        }
    }
    
    public class ProductService
    {
        private List<Product> _products;
        
        public ProductService()
        {
            _products = new List<Product>();
        }
        
        public void AddProduct(Product product)
        {
            _products.Add(product);
        }
        
        public Product GetProductById(int id)
        {
            return _products.Find(p => p.Id == id);
        }
        
        public List<Product> GetAllProducts()
        {
            return _products;
        }
        
        public async Task<List<Product>> GetProductsAsync()
        {
            return await Task.FromResult(_products);
        }
    }
}
