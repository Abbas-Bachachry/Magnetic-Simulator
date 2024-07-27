from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

old_sample_type = 'Clusters'
default_form_data = {'sample_type': 'Clusters', 'packing_factor': 0.001, 'no_cluster': 20,
                     'no_particle_per_cluster': 10,
                     'name': None, 'temperature': 27, 'particle_size': 100, 'box_size': '581.224,581.224,581.224',
                     'no_particle': 150}


class Sample:
    def __init__(self, sample_type, packing_factor=None, no_cluster=None, no_particle_per_cluster=None,
                 no_particle=None, name=None, temperature=None, particle_size=None, box_size=None):
        self.sample_type = sample_type
        self.packing_factor = packing_factor
        self.no_cluster = no_cluster
        self.no_particle_per_cluster = no_particle_per_cluster
        self.no_particle = no_particle
        self.name = name
        self.temperature = temperature
        self.particle_size = particle_size
        self.box_size = box_size
        self.positions = self.generate_positions()

    def generate_positions(self):
        import random
        positions = []
        if self.sample_type == 'Clusters':
            for _ in range(int(self.no_cluster)):
                for _ in range(int(self.no_particle_per_cluster)):
                    positions.append([
                        random.uniform(-1, 1),
                        random.uniform(-1, 1),
                        random.uniform(-1, 1)
                    ])
        elif self.sample_type == 'Random':
            for _ in range(int(self.no_particle)):
                positions.append([
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-1, 1)
                ])
        return positions

    def to_dict(self):
        return {
            'sample_type': self.sample_type,
            'packing_factor': self.packing_factor,
            'no_cluster': self.no_cluster,
            'no_particle_per_cluster': self.no_particle_per_cluster,
            'no_particle': self.no_particle,
            'name': self.name,
            'temperature': self.temperature,
            'particle_size': self.particle_size,
            'box_size': self.box_size,
            'positions': self.positions
        }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = request.form.to_dict()
        sample_type = form_data.get("sample_type")
        sample = Sample(
            sample_type=sample_type,
            packing_factor=form_data.get("packing_factor"),
            no_cluster=form_data.get("no_cluster"),
            no_particle_per_cluster=form_data.get("no_particle_per_cluster"),
            no_particle=form_data.get("no_particle"),
            name=form_data.get("name"),
            temperature=form_data.get("temperature"),
            particle_size=form_data.get("particle_size"),
            box_size=form_data.get("box_size")
        )
        return jsonify(sample.to_dict())
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
