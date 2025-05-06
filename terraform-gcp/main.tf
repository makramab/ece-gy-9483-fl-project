terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0"
    }
  }
}

provider "google" {
  project = "<CHANGE_TO_YOUR_PROJECT_NAME>"
  region  = "us-east1"
  # credentials via ADC (gcloud auth application-default)
}

data "google_compute_image" "ubuntu2404" {
  project = "ubuntu-os-cloud"
  family  = "ubuntu-2404-lts-amd64"
}

# ALLOW ALL INGRESS (not secure—experimentation only!)
resource "google_compute_firewall" "allow_all_ingress" {
  name    = "allow-all-ingress"
  network = "default"

  direction     = "INGRESS"
  priority      = 1000
  source_ranges = ["0.0.0.0/0"]

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "icmp"
  }
}

# ALLOW ALL EGRESS (default in GCP, but shown here for completeness)
resource "google_compute_firewall" "allow_all_egress" {
  name              = "allow-all-egress"
  network           = "default"
  direction         = "EGRESS"
  destination_ranges = ["0.0.0.0/0"]

  allow {
    protocol = "all"
  }
}

# SSH key variables
variable "ssh_username" {
  type    = string
  default = "<CHANGE_TO_YOUR_DESIRED_SSH_USERNAME>"
}
variable "ssh_public_key_path" {
  type    = string
  # Terraform won’t expand '~'—use full path:
  default = "<CHANGE_TO_PATH_TO_YOUR_SSH_KEY>"
}

# 4️⃣ SERVER instance
resource "google_compute_instance" "fl_server" {
  name         = "fl-server-vm"
  machine_type = "e2-standard-4"
  zone         = "us-east1-b"

  boot_disk {
    initialize_params { 
      image = data.google_compute_image.ubuntu2404.self_link 
      size  = 50        # bump root disk to 50 GB
    }
  }

  network_interface {
    network       = "default"
    access_config {}  # external IP
  }

  metadata = {
    ssh-keys = "${var.ssh_username}:${file(var.ssh_public_key_path)}"
  }
}

# CLIENT instances (count = 2)
locals {
  # you can pick any two zones here
  zones = ["us-east1-b", "us-west1-b"]
}

resource "google_compute_instance" "fl_client" {
  count        = 2
  name         = "fl-client-vm-${count.index + 1}"
  machine_type = "e2-standard-4"
  zone         = local.zones[count.index]

  boot_disk {
    initialize_params { 
      image = data.google_compute_image.ubuntu2404.self_link 
      size  = 50        # bump root disk to 50 GB
    }
  }

  network_interface {
    network       = "default"
    access_config {}  # external IP
  }

  metadata = {
    ssh-keys = "${var.ssh_username}:${file(var.ssh_public_key_path)}"
  }
}
