variable "VERSION" {
  default = "1.0.0"
}

variable "REPO_NAME" {
  default = "base-miner-mesh"
}

variable "CACHE_FROM" {
  default = ""
}

variable "CACHE_TO" {
  default = ""
}

group "default" {
  targets = ["gen404-image"]
}

target "gen404-image" {
  platforms  = ["linux/amd64"]
  context = "."
  dockerfile = "docker/Dockerfile"
  cache-from = split(";", CACHE_FROM)
  cache-to = split(";", CACHE_TO)
  tags = [
    "europe-west3-docker.pkg.dev/gen-456515/${REPO_NAME}/${REPO_NAME}:${VERSION}",
  ]
}