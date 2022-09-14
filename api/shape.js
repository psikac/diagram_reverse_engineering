class Shape {
    id;
    shape;
    endPoints;
    connections;

    constructor(id, shape, endPoints) {
        this.id = id;
        this.shape = shape;
        this.endPoints = endPoints;
        this.connections = [];
    }
}