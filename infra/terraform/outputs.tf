output "droplet_ip" {
  description = "Public IP address of the server"
  value       = digitalocean_droplet.app.ipv4_address
}

output "app_url" {
  description = "URL to access the application"
  value       = var.domain != "" ? "http://${var.domain}" : "http://${digitalocean_droplet.app.ipv4_address}"
}

output "ssh_command" {
  description = "SSH command to connect to the server"
  value       = "ssh root@${digitalocean_droplet.app.ipv4_address}"
}
