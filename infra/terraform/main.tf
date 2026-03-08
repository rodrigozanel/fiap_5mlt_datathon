# =============================================================================
# DigitalOcean Droplet for Passos Magicos ML Platform
# =============================================================================

resource "digitalocean_droplet" "app" {
  name     = "${var.project_name}-server"
  region   = var.region
  size     = var.droplet_size
  image    = "docker-20-04"
  ssh_keys = [var.ssh_key_fingerprint]

  tags = [var.project_name, "ml", "production"]

  user_data = templatefile("${path.module}/cloud-init.yaml", {
    repo_url       = var.repo_url
    repo_branch    = var.repo_branch
    jwt_secret_key = var.jwt_secret_key
    api_username   = var.api_username
    api_password   = var.api_password
  })

  connection {
    type = "ssh"
    host = self.ipv4_address
    user = "root"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# =============================================================================
# Firewall
# =============================================================================

resource "digitalocean_firewall" "app" {
  name        = "${var.project_name}-fw"
  droplet_ids = [digitalocean_droplet.app.id]

  # SSH
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  # HTTP
  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  # HTTPS (for future TLS)
  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  # SigNoz UI
  inbound_rule {
    protocol         = "tcp"
    port_range       = "8080"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  # Allow all outbound
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}

# =============================================================================
# Optional: DNS Record
# =============================================================================

resource "digitalocean_domain" "app" {
  count = var.domain != "" ? 1 : 0
  name  = var.domain
}

resource "digitalocean_record" "app_a" {
  count  = var.domain != "" ? 1 : 0
  domain = digitalocean_domain.app[0].id
  type   = "A"
  name   = "@"
  value  = digitalocean_droplet.app.ipv4_address
  ttl    = 300
}
