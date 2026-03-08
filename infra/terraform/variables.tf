variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "ssh_key_fingerprint" {
  description = "Fingerprint of the SSH key already registered in DigitalOcean"
  type        = string
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default     = "nyc1"
}

variable "droplet_size" {
  description = "Droplet size (slug). Minimum s-4vcpu-8gb for full stack"
  type        = string
  default     = "s-4vcpu-8gb"
}

variable "domain" {
  description = "Domain name (optional). Leave empty to use droplet IP"
  type        = string
  default     = ""
}

variable "project_name" {
  description = "Project name for tagging resources"
  type        = string
  default     = "passos-magicos"
}

variable "repo_url" {
  description = "Git repository URL to clone on the server"
  type        = string
  default     = "https://github.com/ricardoandrietta/dataton.git"
}

variable "repo_branch" {
  description = "Git branch to deploy"
  type        = string
  default     = "main"
}

# -- App secrets (passed to .env on the server) --

variable "jwt_secret_key" {
  description = "JWT signing secret for the API"
  type        = string
  sensitive   = true
}

variable "api_username" {
  description = "API login username"
  type        = string
  default     = "admin"
}

variable "api_password" {
  description = "API login password"
  type        = string
  sensitive   = true
}
