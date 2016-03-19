Chart.types.Scatter.extend({
  name: "VideoChart",
  initialize: function(data){
    console.log('My Line chart extension');
    Chart.types.Scatter.prototype.initialize.apply(this, arguments);
    this.horX = 0;
  },
  draw: function(){
    Chart.types.Scatter.prototype.draw.apply(this, arguments);
    // TODO
    var padding = 10;
    var x = this.horX;
    var scale = this.scale;
    this.chart.ctx.beginPath();
    this.chart.ctx.moveTo(x, 0 + padding);
    this.chart.ctx.strokeStyle = '#ff0000';
    this.chart.ctx.lineTo(x, this.chart.ctx.canvas.height - 5*padding);
    this.chart.ctx.stroke();
  },
  updateLine: function(x){
    // Erase line and redraw it
    this.horX = x;
    this.draw();
  }
});
